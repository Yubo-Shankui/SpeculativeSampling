// Copy this file, edit the text fields, and pass it to --game-config.
window.SPECULATIVE_SAMPLING_GAME = {
  title: "Robot Arena",
  tagline: "Choose the robot's next move while the world keeps playing.",
  actionLabel: "Choose the next move",
  actionHelp: "Each move plays once, then returns to idle.",
  battleOverText: "Sequence complete",
  restartLabel: "Play again",
  victoryText: "MISSION COMPLETE",
  referenceError: "Unable to load the game reference image",

  // Repeated for every normal action to preserve camera, world, and identity.
  globalPrompt: `A continuous third-person game view of one small red robot in a
  sunlit concrete training arena. Keep the same camera, arena layout, robot
  design, scale, and lighting. Motion is readable and physically continuous.
  No cuts, HUD, text, logos, or watermarks.`,

  idleAction: 0,
  actions: [
    {
      title: "Idle",
      detail: "Wait and look around",
      prompt: `The robot stands in place, shifts its weight, and slowly looks
      from side to side without leaving the center of the arena.`,
    },
    {
      title: "Dash",
      detail: "Move forward, then stop",
      prompt: `The robot leans forward, dashes several steps across the arena,
      decelerates, and returns to a stable standing pose.`,
    },
    {
      title: "Finish",
      detail: "Jump and complete the sequence",
      terminal: true,
      prompt: `The robot crouches, jumps straight upward, and raises both arms
      at the top of the jump.`,
      followups: [
        {
          prompt: `The robot lands firmly, lowers both arms, and faces the
          camera while small pieces of confetti settle around it.`,
        },
        {
          includeGlobal: false,
          prompt: `A continuous third-person game view of the same sunlit
          concrete training arena. The same small red robot stands calmly after
          completing the sequence while the last confetti settles. No further
          action begins. No cuts, HUD, text, logos, or watermarks.`,
        },
      ],
    },
  ],
};
