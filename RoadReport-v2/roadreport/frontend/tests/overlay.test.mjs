import test from 'node:test';
import assert from 'node:assert/strict';
import {boxAtTime} from '../src/components/trackOverlay.mjs';
const track=[{t:1,box:[.1,.2,.2,.1]},{t:1.25,box:[.2,.4,.2,.1]}];
test('interpolates movement in video coordinates',()=>assert.deepEqual(boxAtTime(track,1.125),[.15000000000000002,.30000000000000004,.2,.1]));
test('no boxes before or after track lifetime',()=>{assert.equal(boxAtTime(track,0),null);assert.equal(boxAtTime(track,2),null);});
test('seeking backwards restores correct box',()=>{boxAtTime(track,1.2);assert.deepEqual(boxAtTime(track,1),track[0].box);});
test('does not draw through an occlusion',()=>assert.equal(boxAtTime([track[0],{...track[1],t:3}],2),null));
test('empty observations are safe',()=>assert.equal(boxAtTime([],0),null));
