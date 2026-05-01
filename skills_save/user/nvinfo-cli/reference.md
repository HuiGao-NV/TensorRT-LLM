# NVInfo CLI — Full Command Reference

All commands require authentication. Run `nvinfo login https://nvinfo.nvidia.com` first.

Every command supports `--json` for raw JSON output (useful for piping or when you need field names not shown in the table display).

---

## search

Search across all company data sources (powered by Glean).

```bash
nvinfo search "<query>" [--limit <n>] [--json]
```

| Flag | Required | Default | Description |
|---|---|---|---|
| `query` | yes | — | Search query text (positional argument) |
| `--limit`, `-n` | no | 10 | Results per page |

---

## people search

Find colleagues by name, email, or login.

```bash
nvinfo people search "<query>" [--limit <n>] [--page <n>] [--json]
```

| Flag | Required | Default | Description |
|---|---|---|---|
| `query` | yes | — | Search query (positional argument) |
| `--limit`, `-n` | no | 10 | Results per page |
| `--page` | no | 0 | Page number (0-indexed) |

---

## people profile

Get a full person profile by login, email, or employee ID.

```bash
nvinfo people profile <login> [--json]
```

| Flag | Required | Description |
|---|---|---|
| `login` | yes | Person login, email, or employee ID (positional) |

Returns department, manager chain, location, contact info.

---

## people reports

List direct reports for a manager.

```bash
nvinfo people reports <manager-login> [--limit <n>] [--json]
```

| Flag | Required | Default | Description |
|---|---|---|---|
| `manager-login` | yes | — | Manager login ID (positional) |
| `--limit`, `-n` | no | 50 | Max results |

**Note:** The command name is `reports`, not `direct-reports`.

---

## rooms search

Search for meeting rooms by name or location.

```bash
nvinfo rooms search "<query>" [--limit <n>] [--json]
```

| Flag | Required | Default | Description |
|---|---|---|---|
| `query` | yes | — | Room name search query (positional) |
| `--limit`, `-n` | no | 10 | Max results |

**Important:** The table display shows Name, Location, Capacity, and Bookable status. To get room **email addresses** (needed for `rooms availability` and `rooms book`), use `--json` and look for email fields in the response.

---

## rooms availability

Check room availability for a time range.

```bash
nvinfo rooms availability --start <start> --end <end> [--location <loc>] [--room-email <email>] [--room-emails <emails>] [--building <name>] [--floor <floor>] [--timezone <tz>] [--json]
```

| Flag | Required | Description |
|---|---|---|
| `--start` | yes | Range start (ISO 8601 or HH:MM) |
| `--end` | yes | Range end (ISO 8601 or HH:MM) |
| `--location` | conditional | Location name — required unless `--room-email` or `--room-emails` given |
| `--room-email` | no | Check a single room by email |
| `--room-emails` | no | Comma-separated room emails to check |
| `--building` | no | Filter by building name |
| `--floor` | no | Filter by floor |
| `--timezone` | no | IANA timezone (defaults to local) |

---

## rooms book

Book a meeting room.

```bash
nvinfo rooms book --room-email <email> --date <YYYY-MM-DD> --start <HH:MM> [--duration <minutes>] [--timezone <tz>] [--title "<title>"] [--description "<text>"] [--yes] [--json]
```

| Flag | Required | Default | Description |
|---|---|---|---|
| `--room-email` | yes | — | Room email address (from `rooms search --json`) |
| `--date` | yes | — | Date (YYYY-MM-DD) |
| `--start` | yes | — | Start time (HH:MM) |
| `--duration` | no | 60 | Duration in minutes |
| `--timezone` | no | local tz | IANA timezone (e.g. `America/Los_Angeles`) |
| `--title` | no | "Team meeting" | Meeting subject/title |
| `--description` | no | — | Meeting description |
| `--yes` | no | — | Skip confirmation prompt |

Interactive prompts will fill in missing fields unless `--non-interactive` is set.

---

## desk locations

List bookable desk locations.

```bash
nvinfo desk locations [--search <filter>] [--json]
```

| Flag | Required | Description |
|---|---|---|
| `--search` | no | Filter locations by name |

Returns location objects with `LocID`. Use `LocID` as `locId` in `desk available`.

---

## desk available

Find available desks at a location.

```bash
nvinfo desk available <locId> [--date <YYYY-MM-DD>] [--json]
```

| Flag | Required | Default | Description |
|---|---|---|---|
| `locId` | yes | — | Location ID from `desk locations` (positional) |
| `--date` | no | today | Date to check (YYYY-MM-DD) |

Returns desk objects with `LandmarkID` (number). Use in `desk book`.

---

## desk book

Book a desk.

```bash
nvinfo desk book --landmark-id <id> --date <YYYY-MM-DD> [--start <HH:MM>] [--end <HH:MM>] [--yes] [--json]
```

| Flag | Required | Default | Description |
|---|---|---|---|
| `--landmark-id` | yes | — | Desk landmark ID (number) from `desk available` |
| `--date` | yes | — | Reservation date (YYYY-MM-DD) |
| `--start` | no | — | Start time (HH:MM) — sets custom time block |
| `--end` | no | — | End time (HH:MM) — required when `--start` is set |
| `--yes` | no | — | Skip confirmation prompt |

Omitting `--start` / `--end` books a full business day (08:00–18:00).

---

## desk reservations

List your current and upcoming desk bookings.

```bash
nvinfo desk reservations [--json]
```

---

## desk cancel

Cancel a desk reservation.

```bash
nvinfo desk cancel <reservationId> [--yes]
```

| Flag | Required | Description |
|---|---|---|
| `reservationId` | yes | Reservation ID from `desk reservations` (positional) |
| `--yes` | no | Skip confirmation prompt |

---

## calendar

View upcoming calendar events.

```bash
nvinfo calendar [--days <n>] [--json]
```

| Flag | Required | Default | Description |
|---|---|---|---|
| `--days`, `-d` | no | 7 | Number of days to look ahead |

**Note:** The command is `calendar` (not `calendar events`). It automatically groups events by day.

---

## news global

Get company-wide news articles.

```bash
nvinfo news global [--limit <n>] [--site <namespace>] [--json]
```

| Flag | Required | Default | Description |
|---|---|---|---|
| `--limit`, `-n` | no | 10 | Number of articles |
| `--site` | no | — | SharePoint site namespace override |

---

## news my

Get personalized news based on your interests and activity.

```bash
nvinfo news my [--limit <n>] [--json]
```

| Flag | Required | Default | Description |
|---|---|---|---|
| `--limit`, `-n` | no | 10 | Number of articles |

---

## news videos

Get company video content.

```bash
nvinfo news videos [--limit <n>] [--json]
```

| Flag | Required | Default | Description |
|---|---|---|---|
| `--limit`, `-n` | no | 10 | Number of videos |

---

## tools

Search for internal tools and applications.

```bash
nvinfo tools "<query>" [--limit <n>] [--json]
```

| Flag | Required | Default | Description |
|---|---|---|---|
| `query` | yes | — | Search query (positional) |
| `--limit`, `-n` | no | 10 | Max results |

**Note:** The command is `tools <query>` directly — there is no `tools search` subcommand and no `tools detail` subcommand.

---

## referral jobs

Browse open job positions for referrals.

```bash
nvinfo referral jobs [--keyword "<role>"] [--page <n>] [--count <n>] [--json]
```

| Flag | Required | Default | Description |
|---|---|---|---|
| `--keyword` | no | — | Filter by title, location, or department |
| `--page` | no | 1 | Page number (1-based) |
| `--count`, `-n` | no | 20 | Results per page |

Note the `jobPostingId` for use with `referral submit`.

---

## referral parse-resume

Upload a resume and extract candidate information into a draft JSON.

```bash
nvinfo referral parse-resume <file> [--json]
```

| Flag | Required | Description |
|---|---|---|
| `file` | yes | Path to resume file (PDF, DOCX, DOC, TXT) |

Returns a draft JSON with extracted fields. Edit it to add `relationship`, `comments`, and `jobPostingIds`, then use with `referral submit --draft`.

---

## referral submit

Submit an employee referral. Runs as an interactive wizard by default, or accepts a pre-filled draft JSON.

```bash
# Interactive mode (prompts for all fields):
nvinfo referral submit [--json]

# Draft mode (from parse-resume output):
nvinfo referral submit --draft <path> [--job-ids <csv>] [--relationship <text>] [--comments <text>] [--yes] [--json]
```

| Flag | Required | Description |
|---|---|---|
| `--draft` | no | JSON draft file from `parse-resume` (use `-` for stdin) |
| `--job-ids` | no | Override job posting IDs (comma-separated) |
| `--relationship` | no | Your relationship to candidate |
| `--comments` | no | Why you are referring this person |
| `--yes` | no | Skip confirmation prompt |
| `--non-interactive` | no | No prompts; draft + flags must supply all required fields |

**Note:** There are no `--candidateFirstName` etc. flags. Candidate info comes from the interactive wizard or the `--draft` JSON file.

---

## priorities

Get your priorities and signals from Teams and Outlook.

```bash
nvinfo priorities [--days <n>] [--query "<filter>"] [--json]
```

| Flag | Required | Default | Description |
|---|---|---|---|
| `--days`, `-d` | no | — | Lookback days |
| `--query`, `-q` | no | — | Filter tasks by keyword |

Returns todo items and signals from Microsoft Teams and Outlook.

---

## approvals

Get pending approvals and action items from the Action Center.

```bash
nvinfo approvals [--json]
```

Combined view of ServiceNow tasks and approval requests needing attention.

---

## servicenow

Get your open ServiceNow tickets.

```bash
nvinfo servicenow [--json]
```

**Note:** The command is `servicenow` directly — there is no `summary` subcommand.

---

## docs

Get your personalized recent documents feed from Glean.

```bash
nvinfo docs [--limit <n>] [--json]
```

| Flag | Required | Default | Description |
|---|---|---|---|
| `--limit`, `-n` | no | 15 | Number of documents |

---

## history

List past NVInfo chat conversations, or view a specific transcript.

```bash
# List all conversations:
nvinfo history [--json]

# View a specific conversation:
nvinfo history <id> [--json]
```

| Flag | Required | Description |
|---|---|---|
| `id` | no | Conversation ID to view (positional, optional) |

**Note:** The command is `history` or `history <id>` — there are no `list` or `delete` subcommands.

---

## MCP Server Commands

The CLI can run as an MCP (Model Context Protocol) server for AI agent integration.

### mcp serve

Start MCP server in stdio mode for local AI tools (e.g., Claude Desktop).

```bash
nvinfo mcp serve
```

Runs in foreground, communicating via stdin/stdout.

### mcp start

Start MCP server as a background HTTP service.

```bash
nvinfo mcp start [--port <n>]
```

| Flag | Required | Default | Description |
|---|---|---|---|
| `--port`, `-p` | no | 37537 | HTTP port to listen on |

### mcp stop

Stop the background MCP server.

```bash
nvinfo mcp stop
```

### mcp status

Check if the MCP server is running.

```bash
nvinfo mcp status
```

### mcp urls

Show MCP server endpoint URLs for agent configuration.

```bash
nvinfo mcp urls
```

---

## Authentication Commands

### login

```bash
nvinfo login <server-url> [--device-code] [--code <code>]
# Example: nvinfo login https://nvinfo.nvidia.com
```

| Flag | Required | Description |
|---|---|---|
| `server-url` | no | NVInfo server URL (uses configured URL if omitted) |
| `--device-code` | no | Use device-code flow instead of browser SSO |
| `--code` | no | Provide confirmation code directly (for automation/AI agents) |

Opens a browser for NVIDIA SSO. Use `--device-code` for headless environments.

**Non-interactive mode:** When running without a TTY (e.g., from AI agents),
the CLI exits with code 2 after opening the browser. Complete sign-in in the
browser, then run `nvinfo login --code <CODE>` with the confirmation code shown.

### auth status

```bash
nvinfo auth status
```

Shows current authentication state, including token validity and expiry. Reports whether the access token is expired so the agent knows to re-login before running commands.

### auth logout

```bash
nvinfo auth logout
```

Signs out and clears stored credentials.

### auth configure

Configure authentication settings (set the NVInfo server URL).

```bash
nvinfo auth configure --url <url>
```

| Flag | Required | Description |
|---|---|---|
| `--url` | yes | NVInfo base URL (e.g. https://nvinfo.nvidia.com) |
