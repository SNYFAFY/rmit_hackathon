

https://github.com/user-attachments/assets/55204f85-7a8a-462f-8afd-26e2a92d7d1a

# ClubVault — Dual-Control Expense Desk

![Prototype status](https://img.shields.io/badge/prototype-SIMULATED%20TideCloak-orange)

> **UNLOCKED Hackathon prototype** — mock money and synthetic identities only.

## 1. The possibility

ClubVault lets a student club spend from a shared fund without giving one person
unilateral control. A Treasurer submits an expense, then two distinct Board
Members approve it. Only then is the mock payment marked `EXECUTED`.

This makes a small but important possibility visible: the club can move quickly
without asking everyone to trust a President, Platform Admin, or AI to override
the rules.

## 2. Why the trust constraint matters

The backend checks the user's role on every request, stores one vote per
distinct user, and uses a single `APPROVAL_THRESHOLD = 2` constant. President
and Platform Admin requests to approve or execute return `403`.

The demo currently runs in **simulated TideCloak mode** because the Tide network
is under heavy load and the local realm could not finish `FinalizeWallet` (HTTP
500). The UI visibly says `SIMULATED TideCloak sign-in (Tide network under
load)`; this is not presented as a live sign-in.

## 3. Why TideCloak matters

The simulation uses separate demo accounts and token-like claims
(`sub`, `realm_access.roles`). Every API request passes through
`verifyToken/getUserRole`; changing to `AUTH_MODE=tide` swaps the provider for
Tide's EdDSA JWT + DPoP verification. Live Tide setup was blocked by network
load, so the simulation is explicit and faithful rather than a claim of live
authentication.

| Step | TideCloak would enforce | ClubVault app code | Evidence/status |
|---|---|---|---|
| Sign-in | Tide identity flow and sender-constrained access token | Starts the session and maps the authenticated user to club data | Tide Next.js auth/DPoP playbooks; simulated here |
| Role claims | Realm/client roles appear in verified JWT claims | Reads verified claims and maps them to ClubVault roles | `add-rbac-nextjs`: `realm_access.roles`; simulated claims mirror this |
| Role checks | Token signature and DPoP proof integrity | Checks role on every expense and governance API request; returns `403` | `protect-api-nextjs` / `verify-jwt-server-side`; live verification path exists |
| Approver promotion via IGA quorum | IGA governs role/approver changes; quorum and enclave signing are native where configured | Displays `PENDING` then `ACTIVE` after two distinct simulated admins | `setup-iga-admin-panel`; propagation/enclave UX is not simulated as live |
| Revocation | Tide role/account revocation changes future verified identity/role claims after propagation | Re-checks claims on every request; no cached authorization | Native revocation propagation timing not verified in this offline simulation |
| Expense threshold | No native app-level N-of-M threshold in the selected path | Counts two distinct approvals and marks mock expense `EXECUTED` | Application-enforced; Forseti is the upgrade path |

Trust gap: the backend is trusted to count Layer 1 votes. A root-compromised
backend could tamper with that count. Votes are not signatures over expense
data; we store `sub`, role, and issued-at time. A Forseti contract deployed by
Tide would move threshold enforcement to the ORK network.

## 4. Prototype flow

1. Select **Treasurer** and submit an expense.
2. Select **Board Member A** and approve: it remains `PENDING`.
3. Select **Board Member B** and approve: it becomes `EXECUTED`.
4. Select **President** or **Platform Admin**: the API returns `403` with a
   clear refusal.
5. Open **History** to see who approved what and when.
6. In Layer 2, as Platform Admin A request Board C's approver role: it is
   `PENDING`, not active.
7. Switch to Platform Admin B and approve: it becomes `ACTIVE`; one admin alone
   could not promote the user.

## 5. Setup

```bash
npm install
cp .env.example .env
# Set KC_BOOTSTRAP_ADMIN_PASSWORD only if using Tide bootstrap.
npm run dev
```

The default is safe simulated mode:

```text
AUTH_MODE=mock
NEXT_PUBLIC_AUTH_MODE=mock
```

After TideCloak licensing and realm bootstrap succeeds, export the adapter to
`data/tidecloak.json` (ignored by git), set both modes to `tide`, then rerun the
Tide doctor and test the browser flow in Chrome/Edge with pop-ups allowed.
Never commit `.env`, the adapter export, or real user data.

## Demo

Open `http://localhost:3000/dashboard`. The synthetic identities are
available from the demo selector so the complete dual-control flow can be shown
in about two minutes.

### Two-minute demo script

1. Say: “No single club officer or AI can release money or promote an approver alone.”
2. As Treasurer, submit “Room deposit” for `$42`; it is `PENDING`, waiting for
   `2/2` distinct Board Member approvals.
3. As Board Member A, approve once; point out that it remains `PENDING`.
4. As President, click approve; show the explicit `403` refusal.
5. As Platform Admin, click approve; show the same refusal.
6. As Board Member B, approve; show `EXECUTED`.
7. Open History and point out the two synthetic `sub` values, roles, and
   timestamps. Explain that these are votes, not signatures over expense data.
8. In Layer 2, have Platform Admin A request Board C's role (`PENDING`), then
   Platform Admin B approve (`ACTIVE`).
9. Close with the Tide boundary: Tide protects identity, role claims, DPoP, and
   IGA promotion; Forseti would move the spending threshold from this backend
   to the ORK network.
