import {nextFraming,INITIAL_FRAMING,LOST_MS,READY_FRAMES} from "../src/framing.js";
import assert from "node:assert";
const full=Array(33).fill(0.9), nohead=full.map((v,i)=>i===0?0.1:v), none=[], torsoGone=Array(33).fill(0.1);
let s=INITIAL_FRAMING, t=0;
for(let i=0;i<READY_FRAMES;i++) s=nextFraming(s,full,t+=33);
assert(s.ready);
for(let i=0;i<200;i++) s=nextFraming(s,nohead,t+=33); // headless for 6s: stay ready
assert(s.ready,"head loss must not re-show");
s=nextFraming(s,none,t+=33); assert(s.ready);
s=nextFraming(s,none,t+=LOST_MS-100); assert(s.ready);
s=nextFraming(s,torsoGone,t+=200); assert(!s.ready,"gross loss re-shows");
s=nextFraming(s,none,t+=33); s=nextFraming(s,full,t+=33); assert(!s.ready);
console.log("framing ok");
