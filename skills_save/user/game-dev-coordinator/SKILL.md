---
name: game-dev-coordinator
description: Coordinate the full game development process for the Three Kingdoms sandbox game. Use when the user wants to plan features, implement new game content, or manage the development workflow. Triggers on "develop", "implement feature", "add to game", "game dev", "build feature", "new feature", "design and build", or when the user describes a game feature request that requires multiple roles.
---

# Game Development Coordinator

Coordinate multi-role game development with **review loops** for the Three Kingdoms sandbox game (Python + Pygame).

## When To Trigger
- User requests a new game feature or system
- User asks to design and implement something
- User wants to understand what work is needed for a change
- User says "develop", "implement", "build", "add feature"
- User describes a game feature that touches multiple systems

## Workflow Overview

```
Step 1: Understand Requirements
Step 2: Identify Roles + Reviewers
Step 3: Create Development Plan
Step 4: Execute with Review Loops
        ┌─────────────────────────────────┐
        │  Role produces deliverable      │
        │         ↓                       │
        │  Reviewer evaluates             │
        │         ↓                       │
        │  APPROVE? ──Yes──→ Next phase   │
        │     │                           │
        │    No (REVISE)                  │
        │     │                           │
        │  Role addresses feedback        │
        │     │                           │
        │  (repeat, max 5 rounds)         │
        │     │                           │
        │  Round 5: auto-approve if       │
        │  no BLOCKERs remain             │
        └─────────────────────────────────┘
Step 5: Integration Review
Step 6: Present Results
```

---

### Step 1: Understand Requirements

Parse the user's request and clarify:
- **What** is the feature/change?
- **Which systems** does it touch? (map, entities, combat, UI, economy, waves, etc.)
- **Scope**: small tweak vs. new system vs. major rework

If the request is ambiguous, ask clarifying questions. Always read existing project files before planning.

Reference the project structure:

```
game/constants.py        - All config: screen, tiles, colors, factions, unit stats
game/game_engine.py      - State machine, main loop
game/camera.py           - Viewport (currently fixed, no scroll)
game/states/
  menu_state.py          - Main menu
  select_state.py        - Faction + path mode selection
  play_state.py          - Core gameplay: draw_path phase + battle phase
  gameover_state.py      - Death screen
game/entities/
  entity.py              - Base class: position, HP, attack, draw shapes/sprites
  lord.py                - Player character (WASD in action mode)
  soldier.py             - Deployable units, movable, no attack while moving
  enemy.py               - Walk along assigned path, attack nearby allies
  projectile.py          - Arrow class for archer projectiles
game/systems/
  map_system.py          - Single/dual-path tile map, path drawing
  wave_system.py         - Multi-wave overlapping spawner
  economy_system.py      - Gold tracking, faction discounts
  combat_system.py       - Range-based auto-attack resolution
  deploy_system.py       - Soldier placement on grass tiles
game/ui/
  hud.py                 - Gold, wave, HP, mode indicator
  shop_panel.py          - Soldier recruitment panel
  button.py              - Reusable button widget
game/sprite_loader.py    - PNG sprite loading with cache
assets/sprites/          - 9 cartoon character PNGs (64x64)
```

---

### Step 2: Identify Required Roles + Reviewers

Every role has a corresponding reviewer. Both are launched as Agent subagents.

| Role | Agent | Reviewer Agent | When Needed |
|------|-------|---------------|-------------|
| Producer | `producer.md` | `producer-reviewer.md` | Task breakdown, schedule |
| Game Designer | `game-designer.md` | `game-designer-reviewer.md` | Mechanics, balance |
| Narrative Designer | `narrative-designer.md` | `narrative-designer-reviewer.md` | Story, lore, text |
| Concept Artist | `concept-artist.md` | `concept-artist-reviewer.md` | Visual design |
| UI/UX Designer | `ui-ux-designer.md` | `ui-ux-designer-reviewer.md` | HUD, menus |
| Lead Programmer | `lead-programmer.md` | `lead-programmer-reviewer.md` | Architecture |
| Gameplay Programmer | `gameplay-programmer.md` | `gameplay-programmer-reviewer.md` | Code implementation |
| Systems Programmer | `systems-programmer.md` | `systems-programmer-reviewer.md` | Engine, perf |
| AI Programmer | `ai-programmer.md` | `ai-programmer-reviewer.md` | Enemy AI, waves |
| Level Designer | `level-designer.md` | `level-designer-reviewer.md` | Map layout |
| Sound Designer | `sound-designer.md` | `sound-designer-reviewer.md` | Audio design |
| QA Lead | `qa-lead.md` | `qa-lead-reviewer.md` | Testing |
| Technical Artist | `technical-artist.md` | `technical-artist-reviewer.md` | Visual effects |

**Minimum roles for common tasks (with reviewers):**
- **Balance tweak**: Game Designer ↔ Reviewer → Gameplay Programmer ↔ Reviewer → QA Lead
- **New unit type**: Game Designer ↔ R → Concept Artist ↔ R → Gameplay Programmer ↔ R → QA Lead
- **New game system**: Producer ↔ R → Game Designer ↔ R → Lead Programmer ↔ R → Gameplay Programmer ↔ R → QA Lead
- **UI change**: UI/UX Designer ↔ R → Gameplay Programmer ↔ R → QA Lead
- **Bug fix**: QA Lead → Gameplay Programmer ↔ R → QA Lead (reviewer verifies fix)

---

### Step 3: Create Development Plan

Generate a structured plan that includes review phases:

```
=== DEVELOPMENT PLAN ===
Feature: [name]
Requested by: User
Complexity: [Low/Medium/High]
Review Mode: [Full (all roles reviewed) / Light (code review only) / Skip (trivial fix)]

Roles Involved:
  1. [Role] ↔ [Role] Reviewer — [task]
  2. [Role] ↔ [Role] Reviewer — [task]

Task Sequence:
  Phase A — Design + Review:
    - [ ] [Role] produces [deliverable]
    - [ ] [Role] Reviewer evaluates (max 5 rounds)
  Phase B — Implement + Review:
    - [ ] [Role] implements code
    - [ ] [Role] Reviewer code review (max 5 rounds)
  Phase C — Integration + Final Review:
    - [ ] Integration checks
    - [ ] QA Lead tests

Estimated files to modify: [list]

Constraints:
  - Chinese text: FONT_PATH   - Colors: CB_* palette only
  - Sprites: assets/sprites/ PNG or pygame.draw fallback
  - Map: 30x20, TILE_SIZE=36, screen 1280x720
========================
```

**Review Mode Selection:**
- **Full**: Complexity=High or new system. Every role's output gets reviewed.
- **Light**: Complexity=Medium. Only code changes get reviewed (gameplay-programmer-reviewer).
- **Skip**: Complexity=Low (typo fix, single value change). No review needed.

Present this plan to the user for approval before executing.

---

### Step 4: Execute with Review Loops

For each task in the plan, run this loop:

#### 4a. Role Produces Deliverable
Launch the role's Agent subagent with:
- The specific task
- Relevant context from previous phases
- Files to read/modify
- Expected HANDOVER REPORT format

#### 4b. Reviewer Evaluates (Review Loop)

```
for round in 1..5:
    Launch reviewer agent with:
      - The role's deliverable/output
      - Original task requirements
      - Previous review feedback (if round > 1)
      - "This is review round {round}/5"

    if reviewer verdict == APPROVE:
        → Move to next task/phase
        break

    elif reviewer verdict == REVISE:
        Launch role agent again with:
          - Reviewer's feedback
          - Specific items to address
          - "Address the BLOCKER and IMPORTANT issues. Round {round}/5"

    elif reviewer verdict == BLOCKED:
        → Escalate to user for decision
        break

if round == 5 and not approved:
    → Auto-approve if no BLOCKERs remain
    → Note unresolved IMPORTANT items in final report
```

#### 4c. Parallel Review Execution

When multiple roles work in parallel, their reviews also run in parallel:
- Game Designer ↔ Reviewer and Concept Artist ↔ Reviewer can review simultaneously
- Only wait for ALL reviews to complete before moving to the next phase

#### 4d. Passing Context Between Phases

When a role's deliverable is approved, extract the key outputs and pass them to the next phase:
- Design specs → Implementation prompt
- Code changes → QA test prompt
- Review feedback that affects other roles → Forward to those roles

---

### Step 5: Integration Review

After all role+review loops complete:
1. **Syntax check**: `python3 -c "import ast; ..."` on all modified .py files
2. **Import check**: Verify no broken imports or missing constants
3. **Consistency**: Colors use CB_* palette, fonts use FONT_PATH, sprites load correctly
4. **No regressions**: Key features still work (state transitions, paths, soldier movement)
5. **Review compliance**: All BLOCKER items resolved, IMPORTANT items addressed or documented

---

### Step 6: Present Results

```
=== FEATURE COMPLETE ===
Feature: [name]
Roles Used: [list]
Review Rounds: [role: N rounds each]
Files Modified: [list with brief description]
Files Created: [list if any]
Testing: [what was verified]

Review Summary:
  - [Role]: Approved in round [N]/5
  - [Role]: Approved in round [N]/5

Unresolved Items: [any IMPORTANT items accepted as trade-offs]
Known Limitations: [if any]
Suggested Follow-ups: [if any]
=========================
```

---

## Project Design Constraints (Always Enforce)

1. **Sprites first, shapes fallback** — entities load PNG from assets/sprites/, fall back to pygame.draw if missing
2. **Colorblind safe** — Use only CB_BLUE, CB_ORANGE, CB_CYAN, CB_VERMILLION, CB_YELLOW, CB_PURPLE from constants.py
3. **Chinese rendering** — All text uses `pygame.font.Font(FONT_PATH, size)`, never SysFont
4. **Screen layout** — Game area left (MAP_COLS * TILE_SIZE), shop panel right (SHOP_PANEL_WIDTH=200)
5. **Single/Dual paths** — Map supports 1 or 2 paths, selected at game start
6. **Soldier movement** — Straight lines (pixel-based), cannot attack while moving
7. **Wave overlap** — Multiple waves can be active simultaneously

## Review Escalation Protocol

If a review loop gets stuck (reviewer and role disagree after 3 rounds):
1. Summarize both positions to the user
2. Ask the user to decide
3. Implement the user's decision
4. Mark as APPROVE with user override noted
