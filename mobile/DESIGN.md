# Design System — CLEAR (mobile)

The visual and interaction system for the CLEAR mobile app. **Read this before making any UI change.** If something here is wrong, fix it here first, then change the code.

A live HTML mockup of every screen described below lives at [`mobile/design-preview.html`](./design-preview.html). Open it in any browser to see the system rendered: 9 phone screens, type specimens, color swatches, spacing, components. This document is the spec; the HTML is the visual reference.

```bash
open mobile/design-preview.html
```

---

## Product context

- **What it is:** a mobile app that classifies skin-lesion photos into broad categories.
- **Who it's for:** people curious about a mark on their skin who want a fast, honest, second opinion before deciding whether to see a dermatologist.
- **Project type:** mobile app (React Native + Expo + TypeScript). No marketing surface. No tablet/web companion.
- **Posture:** product, not a science fair. The app is experimental in its predictions but should feel *finished* as a product.

---

## Design direction

**Aesthetic:** Clinical-warm utility — restrained and functional like a real health tool, but warm earth tones instead of sterile blue/white. Reference: the Claude mobile app's tone (warm cream, single accent, serif/sans pair, considered spacing). NOT Headspace-playful, NOT One-Medical-marketing-polished.

**Memorable thing:** *"Modern health tech that isn't cold."*

**Decoration level:** Minimal. Typography and one accent color do the work. No gradients, no decorative blobs, no illustrations beyond the optional onboarding mark.

**References that informed the system:**
- Claude mobile app — the warm cream + serif/sans + single accent foundation
- One Medical — warm tones over clinical blue, serif display
- Levels Health — modern data presentation in a warm system

---

## Tokens

These values are the system. The mockup CSS is generated from these — when they change here, regenerate the mockup.

### Color

| Token            | Value                       | Usage                                                            |
| ---------------- | --------------------------- | ---------------------------------------------------------------- |
| `bg`             | `#F0EEE6`                   | App background. Warm cream — never pure white.                   |
| `surface`        | `#FAF9F5`                   | Lifted cards, inputs, list rows                                  |
| `surface-2`      | `#FFFFFF`                   | Top-most surfaces if needed (rare; prefer `surface`)             |
| `text`           | `#191919`                   | Primary copy, headlines                                          |
| `muted`          | `#6B6B6B`                   | Secondary copy, metadata, disabled                               |
| `line`           | `rgba(25,25,25,0.08)`       | Hairline borders                                                 |
| `line-strong`    | `rgba(25,25,25,0.14)`       | Toggle track, sheet handle                                       |
| `accent`         | `#D97757`                   | Primary actions, "needs attention" states. Anthropic burnt clay. |
| `accent-pressed` | `#C56843`                   | Pressed/active state on accent buttons and tags                  |
| `accent-tint`    | `rgba(217,119,87,0.1)`      | Tinted backgrounds for accent tags, confidence-dot halo          |
| `calm`           | `#3D5A48`                   | "Low concern" tags only. Sage. Used to differentiate from accent.|
| `calm-tint`      | `rgba(74,107,87,0.1)`       | Background for `calm` tags                                       |
| `error`          | `#A03E3E`                   | App errors and destructive actions only. **Never** predictions.  |

**Rules:**
1. **Color is supportive, not load-bearing.** Every state must also have text or shape that conveys the meaning. A tag still reads correctly to someone with red/green color blindness because the label says "Closer look" or "Low concern."
2. **Predictions never use `error`.** A "suspicious" prediction uses `accent` (terracotta), not red. Red is reserved for app malfunctions and destructive UI.
3. **No pure white as background.** The system commits to warm cream throughout — pure white surfaces look mismatched in this palette.

### Typography

| Role             | Font            | Notes                                                           |
| ---------------- | --------------- | --------------------------------------------------------------- |
| Display / Voice  | Source Serif 4  | Wordmark, "Result" label, screen-title labels, disclaimer copy. **Used sparingly.** This is the app's voice. |
| Body / UI        | Geist           | Headlines, buttons, body text, labels, navigation, everything else |
| Tabular          | Geist (`tnum`)  | Confidence values, dates, percentages, version strings           |

**Banned:** Inter (overused), Roboto, Helvetica/Arial, Open Sans, Lato, Montserrat, Poppins, Space Grotesk. If a system font fallback is needed, use `-apple-system` (iOS) and `Roboto` (Android default) — never set Inter explicitly.

**Why these two:**
- *Source Serif 4* is the closest free alternative to Tiempos (used by Anthropic). It's earnest, contemporary, and reads as "human voice" without feeling old.
- *Geist* is Vercel's alternative to Söhne. Clean, tabular-friendly, no quirks. Avoids the Inter convergence trap.

**Type scale** (in `pt` — RN treats `fontSize` as DP/PT):

| Token          | Size | Weight | Usage                              |
| -------------- | ---- | ------ | ---------------------------------- |
| `display-xl`   | 48   | 500    | Onboarding wordmark, hero pages    |
| `display-lg`   | 30   | 500    | Onboarding screen titles (serif)   |
| `display-md`   | 22   | 500    | Top-bar wordmark (serif)           |
| `headline`     | 28   | 500    | Result headline ("Needs a closer look") |
| `title`        | 18   | 500    | Section titles                     |
| `body`         | 15   | 400    | Default body, list rows            |
| `caption`      | 14   | 400    | Metadata, helper text              |
| `micro`        | 12   | 500    | Section caps, tags, chips          |
| `serif-quote`  | 14.5 | 400    | Body inside info sheets, About paragraphs (serif) |
| `serif-italic` | 14   | 400 *italic* | Disclaimer copy, poetic muted lines (serif italic) |

**Letter-spacing:** -0.02em on serif headlines, -0.01em on sans headlines, +0.08em on uppercase section caps. Body text gets default tracking.

### Spacing

Base unit **4px**. Scale: `4 · 8 · 16 · 24 · 32 · 48 · 64`. Density target: **comfortable** (not compact, not luxurious).

- Screen padding: 24px horizontal, 24px top, 40px bottom (the extra bottom plays nice with the scroll fade).
- Vertical gap inside body: 16px between blocks.
- List row internal padding: 14×16.
- Touch targets: never below 44pt.

### Border radius

| Token       | Value | Usage                            |
| ----------- | ----- | -------------------------------- |
| `r-sm`      | 8     | Inputs, thumbnails               |
| `r-md`      | 12    | Cards, list containers           |
| `r-lg`      | 18    | Photos, big media                |
| `r-sheet`   | 24    | Bottom-sheet top corners         |
| `r-pill`    | 999   | Buttons, chips, tags, toggles    |

### Motion

**Minimal-functional.** Animations exist to clarify hierarchy, not to perform.

- **Fade-in** results: 200ms, ease-out
- **Sheet present**: 280ms, spring `{stiffness: 280, damping: 30}` (RN Reanimated)
- **Tap feedback** on buttons: 80ms scale to 0.98, ease-out
- **Haptic** on capture: `Haptics.ImpactFeedbackStyle.Medium`
- **No** scroll-driven animations, **no** decorative bouncing, **no** auto-playing motion.

---

## Components

| Component           | Spec                                                                                                                |
| ------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **Button (primary)** | Pill, accent fill, white label, `r-pill`, padding 14×24, full-width by default                                      |
| **Button (secondary)** | Transparent bg, muted label, no border, full-width by default                                                     |
| **Button (destructive)** | Transparent bg, `error` label, full-width                                                                         |
| **Input**           | Surface bg, `line` border, `r-md`, padding 14×16, body sans, muted placeholder                                       |
| **Filter chip**     | `r-pill`, surface bg + `line` border by default. Active = accent fill, white label.                                  |
| **Tag**             | `r-pill`, tinted bg + tinted-darker label. Two variants: terracotta tint ("Closer look"), sage tint ("Low concern"). |
| **Severity tag**    | Same as tag but with explicit medical-action language: "See a doctor soon" (urgent / accent), "Generally harmless" (calm). Used inside Condition Info sheets. |
| **Toggle**          | 40×24 pill track. Off = `line-strong`. On = `accent`. 20×20 white thumb.                                            |
| **List row**        | 14×16 padding, body sans label left, value or chevron right. Hairline `line` between rows. Container `r-md`.        |
| **Section cap**     | Uppercase, micro-12 sans, `muted`, letter-spacing 0.08em. Sits above each list group.                               |
| **Bottom sheet**    | Top corners `r-sheet`, drag handle (36×4 `line-strong` pill, centered), padded content. Background veil = 40% black.|
| **Stat card**       | Surface bg, `line` border, `r-md`, centered: large sans tabular number, micro-12 uppercase muted caption below.     |
| **Bar list**        | One row per prediction: label left, percent right, full-width track below in `line` color, fill in `accent` (or `calm` for non-suspicious classes). |
| **Confidence dot**  | 8×8 accent dot with 4px accent-tint halo. Sits next to "Confidence 0.74" tabular text.                              |
| **Photo placeholder** | Square aspect-ratio (1:1) on Scan, slightly taller (4:5) on Detail. `r-lg` corners. Always rounded, never square.   |
| **Action list**     | Serif bullets. Used inside Condition Info sheets for "What you should do."                                           |
| **Avatar**          | Circle in `accent-tint`, large serif initial in `accent-pressed`. 88px on Profile, 32px elsewhere if needed.         |

---

## Screens

Phase 1 currently implements `LoginScreen`, `ScanScreen`, and `HistoryScreen`. The remaining screens below are the target product direction and should be added only when their phase calls for them.

### 1. Onboarding (`OnboardingScreen.tsx`) — deferred
First-launch only, three steps. Step 2 is the **only** place the full disclaimer is presented to the user with prominence.
- Top: progress dots (3 dots, active = accent, wide pill).
- Center: optional accent-tinted illustration mark (120×120 circle).
- Body: serif headline + serif-italic description.
- Bottom: primary "Continue" + secondary "Back".

### 2. Login (`LoginScreen.tsx`)
- Center column: serif "CLEAR" wordmark, serif-italic "skin lesion identification" sub.
- Email + password inputs, primary "Sign in", secondary "Create account".
- **No disclaimer footer.** Full disclaimer/onboarding is deferred to Phase 4 UX polish.

### 3. Scan + Result (`ScanScreen.tsx`)
The core loop. Two states: pre-scan (camera/picker) and post-scan (result, shown here).
- Top app bar: serif wordmark left, "History" link right.
- Square captured photo with `r-lg` corners.
- Below: `Result` (serif italic muted), then the headline (sans, `headline`), then `Confidence 0.74` with the confidence dot.
- Phase 1 auto-saves successful predictions. Primary action is "Take photo"; secondary action is "Choose from library"; result state shows "Saved to history".
- **No disclaimer footer.**

### 4. Scan Detail (`ScanDetailScreen.tsx`) — deferred
Opened by tapping a History row.
- Top app bar: "← Back" left, "Share" right.
- Larger photo, then result block (same as Scan).
- "Top predictions" section cap, then bar list of all classes (sorted desc). The top class uses `accent`, all others use `calm` regardless of prediction.
- Metadata key/value rows: Date, Model.
- Destructive "Delete scan" at bottom.

### 5. History (`HistoryScreen.tsx`)
- Top app bar: serif wordmark, "+ Scan" link.
- "History" section label, search input (`r-pill`, surface, "Search scans…").
- Filter chips: All (active by default), Closer look, Low concern.
- List of history rows: thumbnail (40×40 `r-sm`), tag + confidence stack, date right-aligned.

### 6. Condition Info (sheet, not its own screen) — deferred
Triggered by tapping a row in the Scan Detail bar list. Bottom sheet over Detail.
- Sheet handle, condition title (serif), close ×.
- Subtitle muted (e.g. "Malignant skin cancer").
- **Severity tag** — exactly one: urgent (terracotta) or calm (sage).
- Sections in serif body: "What it is", "What you should do" (action list), "Time to resolve on its own".
- Footer secondary button: "Learn more" (links to NHS / Mayo Clinic / equivalent — pick one authoritative source per condition).

**Content per condition:** maintain a static map in `mobile/src/lib/conditions.ts`. Each entry has `{ name, subtitle, severity, whatItIs, whatToDo[], timeToResolve, learnMoreUrl }`. Treat this as content, not config — review with someone who has clinical context before any change.

### 7. Settings (`SettingsScreen.tsx`) — deferred
Three sections: Account, Notifications, About. Each is a `list` container with rows.
- Account: Email (read-only value), Change password (chevron).
- Notifications: Scan reminders (toggle), Product updates (toggle).
- About: Model version (read-only value), View disclaimer (chevron — opens the disclaimer sheet, same component pattern as Condition Info), Privacy policy (chevron).
- Destructive "Sign out" at bottom.
- Serif-italic version footer: `CLEAR v0.1.0` (no "learning project" framing).

### 8. Profile (`ProfileScreen.tsx`) — deferred
- Top app bar: "← Back" left, "Edit" link right.
- Hero: 88×88 avatar circle, email, serif-italic "Joined April 2026".
- 3-stat grid: Scans, Closer look, Low concern.
- Activity: list rows for "Most recent scan" and "First scan".
- Secondary "Export my data".

### 9. About (`AboutScreen.tsx`) — deferred
- Top app bar: "← Back".
- Center: serif `CLEAR` wordmark (large), serif-italic tagline.
- Long-form body in serif: what CLEAR does, what it isn't.
- "System" section cap, list rows: Model, Trained on, Classes, App version.
- Secondary "View source".

---

## Patterns and house rules

### Disclaimer policy
Target product decision: **the full disclaimer appears only once, during Onboarding step 2.** It is not repeated in screen footers. It remains accessible at any time via Settings → "View disclaimer" (which opens the disclaimer sheet — same bottom-sheet pattern as Condition Info).

Implementation status: this flow is not in Phase 1. It is deferred to Phase 4 UX polish; until then, app copy must stay conservative and avoid medical certainty.

This is intentional. Plastering disclaimers on every screen looks medical and untrustworthy; showing it once with weight and making it always-reachable is more honest.

### Confidence treatment
- **On Scan + Result:** text + accent dot, e.g. `● Confidence 0.74`. **No bar.** A bar overclaims precision and looks diagnostic.
- **On Scan Detail:** bar list of *all* class probabilities, top class accent, others calm. This screen is for someone digging in — bars belong here, not on the primary result.
- Always tabular numerals via `fontVariant: ['tabular-nums']`.

### Tag system
| Tag             | Color    | Means                                               |
| --------------- | -------- | --------------------------------------------------- |
| "Closer look"   | accent   | Top class is one of: melanoma, BCC, AK, SCC         |
| "Low concern"   | calm     | Top class is one of: nevus, BKL, DF, vascular       |

Tags must always include the literal label string — never tag-by-color-only.

### Voice — when to use serif
The serif (Source Serif 4) is the **app's voice**. Used in exactly these places:
- The "CLEAR" wordmark
- The `Result` italic-muted label above the headline
- Onboarding screen titles
- The disclaimer copy (italic, on Onboarding step 2 and inside the disclaimer sheet)
- The body inside Condition Info sheets and the About screen
- Italic muted lines used as poetic micro-copy: "Joined April 2026", "skin-lesion identification", "Does not resolve on its own."

Everywhere else: sans (Geist).

### Product framing
The app is a **product**. App-facing copy never calls CLEAR a "learning project". The honest framing is "experimental classification tool." Internal repository docs (PROJECT.md, AGENTS.md) keep their developer framing — that's not user-facing.

---

## Accessibility

- **Touch targets** ≥ 44×44 pt on every interactive element.
- **Contrast:** all text passes WCAG AA at the rendered size. The `muted` token is the floor for text; never lighter.
- **Color is never the only signal.** Tags say "Closer look" / "Low concern". Severity tags say "See a doctor soon" / "Generally harmless".
- **Tabular numerals** on every number so digit width doesn't shift between values.
- **Dynamic Type** (iOS): respect the user's text-size preference. The type scale above is the *base* — clamp to a sensible max so layouts don't explode.
- **Reduce Motion:** disable the sheet spring and result fade when the user has Reduce Motion enabled. Replace with instant transitions.

---

## Implementation notes (React Native)

### Theme object
Stash the tokens in `mobile/src/theme.ts` as a single readonly object. Don't import strings ad-hoc.

```ts
export const theme = {
  colors: {
    bg: '#F0EEE6',
    surface: '#FAF9F5',
    text: '#191919',
    muted: '#6B6B6B',
    line: 'rgba(25,25,25,0.08)',
    lineStrong: 'rgba(25,25,25,0.14)',
    accent: '#D97757',
    accentPressed: '#C56843',
    accentTint: 'rgba(217,119,87,0.1)',
    calm: '#3D5A48',
    calmTint: 'rgba(74,107,87,0.1)',
    error: '#A03E3E',
  },
  radii: { sm: 8, md: 12, lg: 18, sheet: 24, pill: 999 },
  spacing: { xs: 4, sm: 8, md: 16, lg: 24, xl: 32, xxl: 48, xxxl: 64 },
  fonts: { serif: 'SourceSerif4', sans: 'Geist' },
} as const;
```

### Font loading
Use `expo-font`. Both fonts are on Google Fonts and have Expo packages:
```bash
npx expo install expo-font @expo-google-fonts/source-serif-4 @expo-google-fonts/geist
```
Load in the root component before rendering. Show a blank `bg`-colored splash until fonts are ready — don't flash sans.

### Scrolling
Every screen body sits inside a `<ScrollView>`. The mockup demonstrates the visual result (fixed-height frame, scrollable inner content). On the real device the OS supplies the scroll surface — no manual height math.

### Dark mode
**Deferred.** The system targets light mode. A future migration would invert `bg`/`text`, drop saturation 10–15% on `accent`, and re-test contrast — not a recolor of the same palette. Don't half-build it.

---

## Decisions log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-05-08 | Initial system created | Designed against the "modern health tech that isn't cold" direction. Reference: Claude mobile app. Drove research through One Medical and Levels Health. |
| 2026-05-08 | Accent is `#D97757` (Anthropic burnt clay), not sage | User pivoted to Claude-mobile aesthetic mid-consultation; sage was held back as a secondary "calm" hue used only on tags. |
| 2026-05-08 | Disclaimer shown once (Onboarding step 2), deferred until onboarding/settings exist | Repeating warnings on every screen looks alarmist and untrustworthy. One prominent disclaimer + always-reachable via Settings is more honest. |
| 2026-05-08 | App is framed as a product, not a "learning project" | The internal repo can call this a learning project; the user-facing app cannot — it undermines the product's seriousness. |
| 2026-05-08 | Confidence shown as text + dot on primary Result, bar list on Detail | Bars overclaim precision on the main screen. On Detail the user is digging in, so the bar list is appropriate context. |
| 2026-05-08 | Geist over Inter for body | Inter is the AI-design convergence default. Geist is a peer in quality without that signature. |
