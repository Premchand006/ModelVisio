# Shipping the macOS build (maintainer runbook)

Steps to build, test, and publish the ModelVisio `.dmg`, and to verify that the website's
**Download for macOS** button serves it. The guide for users installing it is
[install-macos.md](install-macos.md).

`.dmg` files can only be built on macOS, so the official build always comes from the GitHub Actions
macOS runner. You don't need a Mac to ship. You do need one (or a Mac-owning tester) to try the
result.

## What gets produced

Every release builds one **universal** app (Apple Silicon + Intel) with
`--target universal-apple-darwin`.

| Release asset | Purpose |
|---|---|
| `ModelVisio_<ver>_universal.dmg` | Versioned installer |
| `ModelVisio-macOS-universal.dmg` | Same file under a fixed name. The website and README link to `/releases/latest/download/ModelVisio-macOS-universal.dmg` |
| `ModelVisio_universal.app.tar.gz` + `.sig` | Signed auto-update payload |
| `latest.json` | Update manifest. Needs `darwin-aarch64` and `darwin-x86_64` entries |

How the app is configured (`apps/desktop/src-tauri/tauri.conf.json` → `bundle.macOS`):

- **Signing:** ad-hoc (`signingIdentity: "-"`). Without at least an ad-hoc signature, Apple
  Silicon calls a downloaded app "damaged".
- **Installer window:** `dmg/background.png` (660×400) shows the "drag to Applications" arrow.
- **Native menus:** the macOS app / File / Edit / View / Window menus are built in
  `src-tauri/src/lib.rs` (`build_menu`).

---

## 0. One-time setup

- **Updater signing key.** Check it's present:

  ```bash
  gh secret list   # expect TAURI_SIGNING_PRIVATE_KEY and TAURI_SIGNING_PRIVATE_KEY_PASSWORD
  ```

  Both are already set. Rotation steps are in [apps/desktop/README.md](../apps/desktop/README.md#auto-update).
- **Apple signing secrets:** not needed. Without them the build is ad-hoc signed and users click
  **Open Anyway** once. To remove that step, see [§6](#6-optional-developer-id-signing--notarization).

## 1. Try a build without releasing

1. Push your branch and open a PR. **Desktop Build Check** runs automatically when the PR touches
   `apps/desktop`, `apps/web`, or `packages`. To run it by hand:

   ```bash
   gh workflow run desktop-build-check.yml --ref <your-branch>
   gh run watch "$(gh run list --workflow desktop-build-check.yml --limit 1 --json databaseId -q '.[0].databaseId')"
   ```

2. The macOS job builds the universal app and fails unless the `.app` passes
   `codesign --verify --deep --strict` and its executable contains both `arm64` and `x86_64`.
3. Download the result. It's the `ModelVisio-macOS-universal-dmg` artifact, kept for 14 days:

   ```bash
   gh run download <run-id> -n ModelVisio-macOS-universal-dmg
   ```

   Or download it from the run's page → **Artifacts**. That route goes through a browser, which
   marks the file as downloaded from the internet, so it reproduces a real user's first launch.

## 2. Test on a Mac

Copy the `.dmg` to a Mac, ideally Apple Silicon running macOS 15 or 26, and follow
[install-macos.md](install-macos.md) as a user would. Check:

- [ ] The `.dmg` window shows the background with the arrow, the app icon on the left and
      Applications on the right
- [ ] Drag to Applications works. The first launch is blocked, and **Privacy & Security → Open
      Anyway** gets the app open. (If you see *"is damaged"*, signing is broken: check the
      Verify step's log.)
- [ ] The menu bar shows **ModelVisio · File · Edit · View · Window**. **ModelVisio → About** shows
      the version and copyright.
- [ ] **⌘V pastes** into the AI Copilot key field. ⌘C / ⌘A / ⌘Z work in text fields.
- [ ] **File → Open Model…** (⌘O) opens a model. Drag-and-drop onto the window works too.
- [ ] AI Copilot answers after pasting a Gemini key
- [ ] ⌘W closes the window, ⌘M minimizes, ⌘Q quits
- [ ] Intel Mac, if available: the app launches natively (Activity Monitor → Kind = *Intel* on
      Intel; *Apple* on Apple Silicon)

## 3. Cut the release

From an up-to-date `main` with the changes merged:

```bash
git checkout main && git pull
pnpm release:desktop 1.1.0          # tags desktop-v1.1.0 and pushes the tag
gh run watch "$(gh run list --workflow desktop-release.yml --limit 1 --json databaseId -q '.[0].databaseId')"
```

The version must be `X.Y.Z`, or a numeric pre-release like `1.1.0-1`. The Windows MSI rejects
`-beta`. All four jobs must go green: macOS universal, Windows, and Linux, plus the stable-name
upload steps.

## 4. Check the draft, then publish

```bash
gh release view desktop-v1.1.0 --json isDraft,assets -q '.isDraft, (.assets[].name)'
```

Expect at least:

```
ModelVisio_1.1.0_universal.dmg
ModelVisio-macOS-universal.dmg
ModelVisio_universal.app.tar.gz
ModelVisio_universal.app.tar.gz.sig
ModelVisio_1.1.0_x64-setup.exe
ModelVisio-Windows-x64-setup.exe
ModelVisio-Linux-x86_64.AppImage
latest.json
```

Then confirm that both kinds of Mac will auto-update:

```bash
gh release download desktop-v1.1.0 -p latest.json -O - | jq '.platforms | keys'
# must include "darwin-aarch64" and "darwin-x86_64"
```

Publish it. A draft is invisible to `/releases/latest`, so every download link 404s until you do:

```bash
gh release edit desktop-v1.1.0 --draft=false --latest
```

## 5. Verify the public links

```bash
for f in ModelVisio-macOS-universal.dmg ModelVisio-Windows-x64-setup.exe ModelVisio-Linux-x86_64.AppImage; do
  printf '%-36s ' "$f"
  curl -sIL -o /dev/null -w '%{http_code}  %{size_download}\n' \
    "https://github.com/Premchand006/ModelVisio/releases/latest/download/$f"
done
# each line must show 200
```

Then:

- Deploy the website from `main` (Netlify/Vercel) so the OS-aware download button and the
  install panel go live. The links themselves never change between releases, so later releases
  need no website deploy.
- Visit the site from a Mac. The main button should read **Download for macOS**, and clicking it
  shows the install steps.
- Installed copies of 1.0.0 should show **Update available** on their next launch.

## 6. Optional: Developer ID signing + notarization

This removes the **Open Anyway** step entirely. It requires the Apple Developer Program
(US$99/year).

1. **Certificate.** In Xcode → Settings → Accounts → *Manage Certificates* → **+** →
   **Developer ID Application**. Or create it at developer.apple.com → Certificates.
2. **Export it.** Keychain Access → My Certificates → right-click the *Developer ID Application*
   certificate → **Export** → `.p12`, with a password.
3. **Get the identity string:**

   ```bash
   security find-identity -v -p codesigning
   # → "Developer ID Application: Your Name (TEAMID)"
   ```

4. **App-specific password.** Create one at <https://account.apple.com> → Sign-In and Security →
   App-Specific Passwords. Your Team ID is at <https://developer.apple.com/account> →
   Membership.
5. **Add the secrets:**

   ```bash
   base64 -i DeveloperID.p12 | gh secret set APPLE_CERTIFICATE
   gh secret set APPLE_CERTIFICATE_PASSWORD      # the .p12 export password
   gh secret set APPLE_SIGNING_IDENTITY          # "Developer ID Application: Your Name (TEAMID)"
   gh secret set APPLE_ID                        # your Apple ID email
   gh secret set APPLE_PASSWORD                  # the app-specific password
   gh secret set APPLE_TEAM_ID
   ```

   The release workflow's *Configure Apple signing* step exports these only when they're
   non-empty, and logs whether signing and notarization are on. Tauri treats an *empty* variable
   as set, so never wire the secrets into `env:` directly.
6. **Cut a release (§3) and verify on a Mac:**

   ```bash
   spctl -a -vvv -t install /Applications/ModelVisio.app
   # → accepted, source=Notarized Developer ID
   ```

7. **Remove the first-launch instructions** from `docs/install-macos.md`,
   `packages/core/src/components/MacInstallHelp.tsx`, the release notes in
   `.github/workflows/desktop-release.yml`, the background text in
   `apps/desktop/src-tauri/dmg/background.png`, and `README.md`.

## Troubleshooting CI

| Failure | Cause / fix |
|---|---|
| *TAURI_SIGNING_PRIVATE_KEY … is not a valid Tauri private key* | The secret must be the whole `.key` file contents. See the rotation steps in apps/desktop/README.md. |
| `codesign --verify` fails in *Verify and attach stable-named macOS .dmg* | `bundle.macOS.signingIdentity` was removed, or an empty `APPLE_SIGNING_IDENTITY` leaked into the env. |
| *No .dmg found* / `bundle_dmg.sh` failed | Finder's AppleScript layout occasionally times out on hosted runners. Re-run the failed job (`gh run rerun <id> --failed`). |
| `can't find crate for core` / target not installed | The rust-toolchain step must install `aarch64-apple-darwin,x86_64-apple-darwin` on macOS. |
| `latest.json` lacks a darwin entry | The macOS job failed or didn't upload. Re-run it before publishing. |
| Download link 404 after publishing | The release isn't marked *Latest* (`gh release edit <tag> --latest`), or the stable-name upload step was skipped. |

## Building on a Mac locally (optional)

```bash
xcode-select --install                                        # once
rustup target add aarch64-apple-darwin x86_64-apple-darwin    # once
pnpm install
pnpm --filter @modelvisio/desktop exec tauri build \
  --target universal-apple-darwin --config src-tauri/tauri.ci.conf.json
open apps/desktop/src-tauri/target/universal-apple-darwin/release/bundle/dmg/
```

`tauri.ci.conf.json` turns off updater artifacts, so no signing key is needed locally. A local
build isn't quarantined, so it won't show the Gatekeeper prompt. Test first-launch behaviour with
a CI artifact downloaded through a browser.
