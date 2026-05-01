---
name: nvinfo-cli
description: >-
  Access NVIDIA enterprise tools via the NVInfo CLI — search company data,
  look up people and org charts, book meeting rooms and desks, check
  calendars, browse news, submit referrals, view ServiceNow tickets, and
  more. Use when the user asks about NVIDIA internal information, colleague
  lookup, room or desk booking, calendar, company news, IT tickets, or
  employee referrals.
---

# NVInfo CLI

NVInfo CLI gives you terminal access to NVIDIA enterprise services.
Run commands directly — no browser needed.

## Before Running Any Command

Before executing any `nvinfo` command, check that the CLI is installed
and the user is authenticated. Follow this sequence:

### Step 1 — Verify CLI exists

```bash
nvinfo --version
```

If this fails with "command not found", do NOT attempt to download or
install the CLI automatically. Tell the user:

> "The NVInfo CLI is not installed on this machine. Please ask your IT
> admin to re-run the NVInfo deployment policy, or run `install.sh`
> from the nvinfo-cli-skill distribution package."

Node.js 18+ is required. If the CLI is present but fails due to a
missing Node.js runtime, tell the user:

> "NVInfo CLI requires Node.js 18+. Install it with
> `brew install node@20` (macOS) or from https://nodejs.org, then retry."

### Step 2 — Verify authentication

```bash
nvinfo auth status
```

If the output indicates the user is **not signed in** or the token is
**expired**, trigger login automatically:

```bash
nvinfo login https://nvinfo.nvidia.com
```

This opens a browser window for NVIDIA SSO. Wait for the login to
complete before running the original command. Tell the user:
"Opening your browser to sign in to NVInfo. Please complete the
sign-in and I'll continue."

**Non-interactive mode (AI agents):** When running in a non-TTY
environment, the CLI will exit with code 2 and display instructions.
The user must complete browser sign-in and then run:

```bash
nvinfo login --code <CONFIRMATION_CODE>
```

where `<CONFIRMATION_CODE>` is the code shown in the browser after
signing in (e.g., `BB4B3A4E`).

### Step 3 — Run the requested command

Only after Steps 1-2 succeed, execute the user's request.

If any command returns a `401 Unauthorized` error mid-session, re-run
`nvinfo login https://nvinfo.nvidia.com` and retry the command once.

## IMPORTANT: Confirmation Required Before Any Write Operation

Before executing any command that **creates, modifies, or deletes** data,
you MUST:

1. Summarize exactly what you are about to do in plain language.
2. Ask the user: **"Shall I go ahead?"**
3. Wait for an explicit "yes" or equivalent confirmation.
4. Only then run the command.

**Write commands that require confirmation:**
- `nvinfo desk book` — books a desk reservation
- `nvinfo desk cancel` — cancels an existing reservation
- `nvinfo rooms book` — books a meeting room
- `nvinfo referral submit` — submits an employee referral

**Example — correct flow for desk booking:**

> User: "Book me a desk at SJC19 for Friday"
>
> Claude runs read-only commands to find options, then says:
> "I'm about to book **Desk A42, Floor 3, SJC19** for **Friday March 28**.
> Shall I go ahead?"
>
> User: "Yes"
>
> Claude runs `nvinfo desk book ...`

Never book, submit, or delete on behalf of the user without this
explicit confirmation step.

## Command Reference (Quick)

### Search

| Command | What it does |
|---|---|
| `nvinfo search "<query>"` | Search across all company data (Glean) |

### People

| Command | What it does |
|---|---|
| `nvinfo people search "<name>"` | Find colleagues by name, email, or login |
| `nvinfo people profile <login>` | Full profile (dept, manager, location) |
| `nvinfo people reports <managerLogin>` | List a manager's direct reports |

### Meeting Rooms

Typical workflow: search rooms -> check availability -> **confirm** -> book.

| Command | What it does |
|---|---|
| `nvinfo rooms search "<building>"` | Find rooms by name/location |
| `nvinfo rooms availability --start <iso> --end <iso> --room-emails <emails>` | Check room free slots |
| `nvinfo rooms book --room-email <email> --date <YYYY-MM-DD> --start <HH:MM> --duration <min> --timezone <tz> --title "<title>"` | Book a room (**requires confirmation**) |

**Tip:** Use `nvinfo rooms search "<building>" --json` to get room email
addresses needed for availability checks and booking.

### Desks

Typical workflow: list locations -> find available desks -> **confirm** -> book.

| Command | What it does |
|---|---|
| `nvinfo desk locations` | List bookable floor locations (returns `LocID`) |
| `nvinfo desk available <locId> --date <YYYY-MM-DD>` | Available desks at a location |
| `nvinfo desk book --landmark-id <id> --date <YYYY-MM-DD>` | Book a desk (**requires confirmation**) |
| `nvinfo desk reservations` | Your current bookings |
| `nvinfo desk cancel <reservationId>` | Cancel a booking (**requires confirmation**) |

### Calendar

| Command | What it does |
|---|---|
| `nvinfo calendar` | Calendar events (defaults to next 7 days) |
| `nvinfo calendar --days <n>` | Events for next N days |

### News

| Command | What it does |
|---|---|
| `nvinfo news global` | Company-wide news articles |
| `nvinfo news my` | Personalized news based on your interests |
| `nvinfo news videos` | Company video content |

### Internal Tools

| Command | What it does |
|---|---|
| `nvinfo tools "<query>"` | Search internal tools and portals |

### Employee Referrals

Typical workflow: browse jobs -> (optional) parse resume -> **confirm** -> submit referral.

| Command | What it does |
|---|---|
| `nvinfo referral jobs --keyword "<role>"` | Browse open positions |
| `nvinfo referral parse-resume <file>` | Extract candidate info from a resume file |
| `nvinfo referral submit` | Submit a referral (interactive wizard) |
| `nvinfo referral submit --draft <file.json>` | Submit from a draft JSON (from parse-resume) |

### Priorities & Tasks

| Command | What it does |
|---|---|
| `nvinfo priorities` | Your priorities and signals from Teams and Outlook |
| `nvinfo approvals` | Pending approvals and action items |
| `nvinfo servicenow` | Your open IT tickets |

### Productivity

| Command | What it does |
|---|---|
| `nvinfo docs` | Personalized recent documents feed |
| `nvinfo history` | Past NVInfo conversations |
| `nvinfo history <id>` | View a specific conversation transcript |

## Common Workflows

### "What's on my plate today?"

```bash
nvinfo calendar --days 1
nvinfo priorities
```

### "Book a room for a 1:1 tomorrow at 2pm"

```bash
nvinfo rooms search "SJC19" --json
# Extract room email from JSON results

nvinfo rooms availability --start 2026-03-25T14:00:00 --end 2026-03-25T15:00:00 --room-emails <email-from-search>

# Present results and ask: "I found Room XYZ available. Shall I book it?"
# On confirmation:
nvinfo rooms book --room-email <email> --date 2026-03-25 --start 14:00 --duration 60 --timezone America/Los_Angeles --title "1:1 sync"
```

### "Book me a desk at SJC19 for Friday"

```bash
nvinfo desk locations --search SJC19
nvinfo desk available <locId> --date 2026-03-27

# Present results and ask: "I found Desk A42 available. Shall I book it?"
# On confirmation:
nvinfo desk book --landmark-id <id> --date 2026-03-27
```

### "Who reports to Jane Smith?"

```bash
nvinfo people search "Jane Smith"
nvinfo people reports <login-from-results>
```

### "Submit a referral for someone"

```bash
nvinfo referral jobs --keyword "Senior Engineer"
# Note the job posting ID(s)

# Option A: Interactive wizard (asks all fields)
nvinfo referral submit

# Option B: Parse resume first, then submit from draft
nvinfo referral parse-resume resume.pdf --json > draft.json
# Edit draft.json: add relationship, comments, jobPostingIds
nvinfo referral submit --draft draft.json
```

### MCP Server (AI Agent Integration)

The CLI can run as an MCP (Model Context Protocol) server for AI agent integration.

| Command | What it does |
|---|---|
| `nvinfo mcp serve` | Start MCP server in stdio mode (for local AI tools) |
| `nvinfo mcp start` | Start MCP server as background HTTP service |
| `nvinfo mcp stop` | Stop the background MCP server |
| `nvinfo mcp status` | Check if MCP server is running |
| `nvinfo mcp urls` | Show MCP server endpoint URLs |

### Authentication

| Command | What it does |
|---|---|
| `nvinfo auth status` | Check authentication state and token validity |
| `nvinfo auth login` | Sign in via browser (alias: `nvinfo login`) |
| `nvinfo auth logout` | Sign out and clear stored credentials |
| `nvinfo auth configure` | Configure authentication settings |

## Error Handling

- **`nvinfo: command not found`** — Tell the user to contact IT to re-run the NVInfo deployment policy. Do NOT attempt to install the CLI automatically.
- **`Token expired`** / **`401 Unauthorized`** — Run `nvinfo login https://nvinfo.nvidia.com` and retry.
- **`ECONNREFUSED` / network errors** — User must be on NVIDIA VPN. Tell them to connect and retry.
- **Node.js not found** — Tell user to install Node.js 18+: `brew install node@20` or https://nodejs.org.

## Detailed Reference

For full parameter documentation on all commands, see [reference.md](reference.md).
