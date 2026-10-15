# Create a new interactive game in five minutes

A game is a small JavaScript object containing one global prompt and a list of
actions. The controller, next-prompt queue, video transport, session lifecycle,
and action buttons are shared; defining a game does not require backend or
controller changes.

## 1. Copy the template

```bash
cp examples/games/my_game.js examples/games/your_game.js
```

## 2. Edit four fields

Open `your_game.js` and change:

1. `title` and `tagline`;
2. `globalPrompt`, which fixes the camera, scene, identities, and visual rules;
3. `actions`, each with a short `title`, UI `detail`, and generation `prompt`;
4. `idleAction`, the action used for automatic continuation.

Normal actions automatically run once and then return to `idleAction`. Keep
action prompts focused on visible motion; put repeated world and identity rules
in `globalPrompt`.

## 3. Optional multi-stage or terminal action

Set `terminal: true` to disable further action buttons after an action is
submitted. Add `followups` when the sequence needs more than one native
proposal:

```javascript
{
  title: "Finish",
  detail: "Two-stage ending",
  terminal: true,
  prompt: "The character begins the finishing move.",
  followups: [
    {prompt: "The move connects and the opponent disappears."},
    {
      includeGlobal: false,
      prompt: "A self-contained post-game prompt with no opponent present.",
    },
  ],
}
```

Follow-ups are queued in order. `includeGlobal` defaults to `true`. Use
`false` only for a self-contained state transition—such as a post-battle scene
that must not mention or respawn the defeated opponent.

## 4. Start the service

Use a reference image matching the identities and scene in the global prompt:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  scripts/h3_live_demo.py \
  --model "$H3_MODEL_PATH" \
  --deployment-config speculative_sampling/models/h3/configs/boundary_image_5s_b300_sol_h3_vae.json \
  --token-file results/h3/token \
  --output results/h3/your-game \
  --reference /path/to/your-game-reference.png \
  --game-config examples/games/your_game.js \
  --generation-width 832 --generation-height 480 --proposal-frames 124 \
  --vae-decode stream --rtc-width 832 --rtc-height 480 \
  --bind 127.0.0.1 --port 8865
```

Open `http://127.0.0.1:8865/?game=1#token=YOUR_PRIVATE_TOKEN`.

## Prompt design checklist

- The reference image and `globalPrompt` describe the same scene and identities.
- Every action has preparation, execution, and recovery that fit one proposal.
- Idle is visually stable and safe to repeat indefinitely.
- Nonterminal actions leave the world in a state from which idle makes sense.
- A terminal follow-up uses a self-contained post-game prompt if the global
  prompt mentions an entity that must not return.
- Prompts request no generated HUD, subtitles, logos, or watermarks; add UI in
  HTML instead.

That is the complete game-facing API. More complex input devices can map onto
the same action submissions without changing Speculative Sampling itself.
