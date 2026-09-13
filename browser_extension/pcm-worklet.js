// AudioWorkletProcessor：把标签页音频整理成"定长、交错、float32"的 PCM 块。
//
// 为什么不直接在 offscreen 文档里用 ScriptProcessorNode：
//   1) 它已被废弃，且跑在主线程上（主线程一卡就丢样本）；
//   2) 它的缓冲大小是"设备量子"的倍数，不好精确控制块长。
// AudioWorklet 在音频线程上跑，量子固定 128 帧，我们自己攒成 20ms 一块。

class LstPcmProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const opts = (options && options.processorOptions) || {};
    this.outChannels = Math.max(1, opts.channels || 2);
    // 20ms @48k = 960 帧；这也是 WS 上一条二进制帧的大小（960*2*4 = 7680 字节）
    this.framesPerChunk = Math.max(128, opts.framesPerChunk || 960);
    this.buf = new Float32Array(this.framesPerChunk * this.outChannels);
    this.filled = 0;
    this.received = 0;
    this.calls = 0;
    this.chunks = 0;
    // 停止采集时要把最后不足一块的数据也吐出去（否则尾巴上 0–20ms 会丢）
    this.port.onmessage = (e) => {
      if (e.data && e.data.type === "flush") this.flush();
    };
  }

  process(inputs) {
    this.calls++;
    // 每 ~0.5 秒报一次"我还活着"：这是判断 worklet 有没有被音频线程驱动的唯一办法
    if (this.calls % 200 === 1) {
      this.port.postMessage({
        type: "alive",
        calls: this.calls,
        received: this.received,
        chunks: this.chunks,
        channels: (inputs[0] && inputs[0].length) || 0,
      });
    }
    const input = inputs[0];
    if (!input || input.length === 0 || !input[0]) {
      return true; // 没有输入（例如标签页静音）时保持存活
    }
    const frames = input[0].length;
    this.received += frames;

    for (let i = 0; i < frames; i++) {
      const base = this.filled * this.outChannels;
      for (let c = 0; c < this.outChannels; c++) {
        // 单声道输入复制到两个声道；多声道只取前两个
        const ch = input[Math.min(c, input.length - 1)];
        this.buf[base + c] = ch ? ch[i] : 0;
      }
      this.filled++;
      if (this.filled >= this.framesPerChunk) {
        this.flush();
      }
    }
    return true;
  }

  flush() {
    if (this.filled === 0) return;
    const copy = this.buf.slice(0, this.filled * this.outChannels);
    this.chunks++;
    // transfer 出去（零拷贝）；对方收到的是 ArrayBuffer
    this.port.postMessage(copy.buffer, [copy.buffer]);
    this.filled = 0;
  }
}

registerProcessor("lst-pcm", LstPcmProcessor);
