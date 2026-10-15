const assert = require("node:assert/strict");
const StartupBuffer = require("../speculative_sampling/runtime/web/startup_buffer.js");
let closed = 0;
const packet = (start, end) => ({unit: {start, end}, decoded: [{close: () => closed++}]});
const q = new StartupBuffer(41);
for (const [a, b] of [[0,9],[9,17],[17,25],[25,33]])
  assert.deepEqual(q.push(packet(a,b)), []);
assert.equal(q.frames,33);
assert.equal(q.push(packet(33,41)).length,5);
assert.equal(q.frames,0);
assert.equal(q.started,true);
assert.equal(q.push(packet(41,49)).length,1);
assert.equal(closed,0);
const immediate = new StartupBuffer();
assert.equal(immediate.push(packet(0,8)).length,1); // H3 default unchanged
const stopped = new StartupBuffer(41);
stopped.push(packet(0,9));stopped.push(packet(9,17));stopped.clear();
assert.equal(closed,2);
assert.equal(stopped.frames,0);
assert.throws(() => new StartupBuffer(-1));
assert.throws(() => q.push(packet(1,1)));
console.log("startup buffer: all checks passed");
