# Timeline Mode — AI Prompt Breakdown Instructions

Give these instructions to an AI assistant along with your full video description. It will return properly formatted segments ready to paste into the Timeline Mode UI.

---

## Instructions for AI

When I give you a video description, break it down into segments for my video generation app. Each segment represents one continuous action/scene that will be generated as a ~1-6 second video clip. The clips are chained together into one final video.

### How the system works:

1. I provide ONE reference photo (the starting image)
2. For each segment, the AI image editor (Qwen) EDITS the current frame to match the segment's description
3. Then the video model (Wan) ANIMATES that edited frame for the specified duration
4. The LAST FRAME of each video segment becomes the input image for the NEXT segment's edit
5. All segments are concatenated into one final video

### Rules for writing segment prompts:

- Each segment prompt is an IMAGE EDITING instruction (what should change in the current frame)
- It is NOT a video motion description — motion is handled separately
- Think of it as: "modify this photo so that [your instruction]"
- Keep instructions LITERAL and VISUAL — describe what should be SEEN in the frame
- Reference the previous state: "she now does X" or "he moves to Y position"
- Max 6 seconds per segment (shorter = better quality, 3-5s is ideal)
- Longer actions need more segments (e.g. "walks across room" = 2-3 segments)

### Duration guidelines:

- **1-2 seconds:** Quick transitions, facial expressions, small movements
- **3-4 seconds:** Standard actions (leaning in, changing position, single gesture)
- **5-6 seconds:** Sustained actions (ongoing motion, slow movements, establishing shots)

### Response format:

Return ONLY the segments in this exact format:

```
MOTION PROMPT: [one line describing camera/animation style for ALL segments]

SEG 1 (Xs): [edit instruction for what the frame should look like]
SEG 2 (Xs): [edit instruction]
SEG 3 (Xs): [edit instruction]
...
```

Where X is the duration in seconds (0.5 to 6, use 0.5 increments).

---

## Example

### My input:
"A woman is sitting on a bed. She slowly removes her top, then lays back and spreads her legs"

### AI response:

```
MOTION PROMPT: Smooth cinematic camera movement, natural body motion, soft bedroom lighting, intimate close angle

SEG 1 (4s): The woman reaches up and grabs the bottom of her top with both hands, beginning to lift it upward. Her midriff is now exposed. Same face, same room.
SEG 2 (3.5s): The woman has completely removed her top which is now gone. Her bare breasts are fully exposed. She is still sitting upright on the bed. Same face and body.
SEG 3 (3s): The woman leans backward and lays down flat on her back on the bed. Her breasts are visible from this angle. Same face.
SEG 4 (4s): The woman is laying on her back with her legs now spread apart, knees bent and open. Her full body is visible from between her spread legs. Same face, everything else the same.
```

---

## Another example

### My input:
"Two women kneeling, taking turns giving blowjob every 5 seconds"

### AI response:

```
MOTION PROMPT: Close-up angle, natural head bobbing motion, detailed skin texture, steady camera

SEG 1 (5s): Woman 1 is kneeling with her mouth fully around the penis, lips sealed at the base, deep throat position. Direct eye contact with camera. Woman 2 kneels beside her holding the base with her hand, watching. Heavy drool visible.
SEG 2 (2s): Woman 1 pulls her mouth completely off the penis and leans back. Woman 2 begins leaning forward toward the penis. Both women's faces clearly visible and unchanged.
SEG 3 (5s): Woman 2 now has her mouth fully around the penis in deep throat position, lips at the base. Direct eye contact with camera. Woman 1 kneels beside holding the base, watching. Same faces as before.
SEG 4 (2s): Woman 2 pulls completely off. Woman 1 leans forward again to take over. Both faces visible and unchanged.
SEG 5 (5s): Woman 1 is back in deep throat position, mouth fully around penis, eye contact. Woman 2 holds base and watches. Same exact faces and bodies throughout.
```

---

## Tips for best results:

1. **Be explicit about what stays the same** — add "same face", "everything else unchanged"
2. **Describe the END STATE** of each segment, not the motion (motion comes from the video model)
3. **Short transition segments (1-2s)** between major position changes help continuity
4. **Don't skip steps** — if she needs to move from standing to laying down, add the transition
5. **Camera angle stays consistent** unless you explicitly change it in a segment
6. **The more segments, the better the continuity** — 3-second segments chain better than 6-second ones
