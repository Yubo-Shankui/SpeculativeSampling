// Buffer only startup packets. It never changes q, source spans or PCM.
(function(root) {
  class StartupBuffer {
    constructor(targetFrames = 0) {
      if (!Number.isInteger(targetFrames) || targetFrames < 0)
        throw new Error("startup target must be a nonnegative integer");
      this.targetFrames = targetFrames;
      this.pending = [];
      this.frames = 0;
      this.started = false;
    }
    push(packet) {
      const count = packet.unit.end - packet.unit.start;
      if (!Number.isInteger(count) || count <= 0)
        throw new Error("invalid startup packet interval");
      if (this.started) return [packet];
      this.pending.push(packet);
      this.frames += count;
      if (this.frames < this.targetFrames) return [];
      this.started = true;
      const result = this.pending;
      this.pending = [];
      this.frames = 0;
      return result;
    }
    clear() {
      for (const packet of this.pending)
        for (const bitmap of packet.decoded) bitmap.close();
      this.pending = [];
      this.frames = 0;
    }
  }
  root.H3StartupBuffer = StartupBuffer;
  if (typeof module !== "undefined") module.exports = StartupBuffer;
})(typeof globalThis !== "undefined" ? globalThis : window);
