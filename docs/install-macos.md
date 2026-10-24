# Installing ModelVisio on macOS

This guide covers getting ModelVisio onto a Mac, opening it for the first time, and keeping it up
to date. It takes about two minutes.

**You need:** a Mac running macOS 10.13 High Sierra or later (a recent macOS is recommended).
The app works on both Apple Silicon (M1 to M4) and Intel Macs. There's one download for both, so you don't need to know which chip you have.

---

## 1. Download

Use any of these. They all give you the same file, `ModelVisio-macOS-universal.dmg` (about 10 to 20 MB).

- **Website:** click **Download for macOS** on the ModelVisio home page.
- **Direct link:** <https://github.com/Premchand006/ModelVisio/releases/latest/download/ModelVisio-macOS-universal.dmg>
- **Releases page:** <https://github.com/Premchand006/ModelVisio/releases/latest>. Under **Assets**,
  pick `ModelVisio-macOS-universal.dmg`. `ModelVisio_<version>_universal.dmg` is the same file
  with the version number in its name.

The file lands in your **Downloads** folder.

## 2. Install

1. Double-click `ModelVisio-macOS-universal.dmg` in Downloads. A window opens showing the ModelVisio
   icon, an arrow, and your **Applications** folder.
2. **Drag the ModelVisio icon onto the Applications folder.** Wait for the copy to finish.
3. Close the window, then eject the disk image: click the ⏏ next to **ModelVisio** in the Finder
   sidebar, or drag its desktop icon to the Trash.
4. You can now delete the `.dmg` from Downloads.

> Always run ModelVisio from **Applications**, not from inside the disk image window. Run from the
> disk image, macOS starts it from a read-only copy, so it can't save settings or update itself.

## 3. Open it for the first time

ModelVisio isn't notarized by Apple yet, so macOS blocks it the **first time** you open it. You
approve it once, and after that it opens normally. Follow the steps for your macOS version. To
find your version, open the Apple menu  → **About This Mac**.

### macOS 15 Sequoia and later

1. Open **Applications** and double-click **ModelVisio**.
2. macOS shows a message titled *"ModelVisio" Not Opened* that says *Apple could not verify
   "ModelVisio" is free of malware…*. Click **Done**. Do **not** click *Move to Trash*.
3. Open the Apple menu  → **System Settings** → **Privacy & Security**.
4. Scroll down to the **Security** section. You'll see *"ModelVisio" was blocked to protect your
   Mac.* Click **Open Anyway**.
5. Enter your Mac password or use Touch ID.
6. A final dialog asks whether to open it. Click **Open Anyway**.

The **Open Anyway** button only appears for about an hour after you try to open the app. If it's
missing, repeat step 1 and look again.

### macOS 14 Sonoma and earlier

1. Open **Applications**.
2. **Control-click** (or right-click) **ModelVisio** and choose **Open**.
3. In the dialog, click **Open**.

If you double-clicked first and got *"can't be opened because Apple cannot check it for
malicious software"*, click **OK**. Then either do the Control-click steps above or use
**System Settings → Privacy & Security → Open Anyway**.

### Alternative: one Terminal command (any macOS version)

If you're comfortable with Terminal, this removes the download block in one step and works on
every version. Open **Terminal** (Spotlight: ⌘Space, type *Terminal*), paste:

```bash
xattr -dr com.apple.quarantine /Applications/ModelVisio.app
```

press Return, then open ModelVisio normally. It produces no output when it succeeds.

## 4. Start using it

- **Launch:** from Applications, Launchpad, or Spotlight (⌘Space → *ModelVisio*). To keep it in the
  Dock, right-click its Dock icon → **Options → Keep in Dock**.
- **Open a model:** **File → Open Model…** (⌘O), drag a model file onto the window, or click
  **Browse Files**. ONNX, TFLite, PyTorch, Keras, Core ML, OpenVINO, GGUF, Safetensors and many
  more formats are supported.
- **AI Copilot (optional):** it uses your own free Gemini API key.
  1. Get a key at <https://aistudio.google.com/apikey>.
  2. In ModelVisio, open the **AI Copilot** panel and click the **key icon**.
  3. Paste the key with ⌘V.

  The key is stored only on your Mac.
- **Standard Mac shortcuts work:** ⌘C / ⌘V / ⌘X / ⌘A / ⌘Z in text fields, ⌘W to close the window,
  ⌘M to minimize, ⌃⌘F for full screen, ⌘H to hide, ⌘Q to quit.

## 5. Updates

ModelVisio checks for updates each time it starts. When a new version is out you'll see
**Update available**. Click **Update now** and the app downloads the update, installs it, and
restarts. Or click **Later**, and you'll be asked again next launch. Updates are cryptographically
verified before they install.

To update by hand instead, download the latest `.dmg` (step 1), drag ModelVisio onto
Applications again, and choose **Replace** when asked.

## 6. Troubleshooting

| What you see | Fix |
|---|---|
| *"ModelVisio" is damaged and can't be opened. You should move it to the Trash.* | The app isn't damaged; macOS shows this for some downloaded apps. Run the Terminal command from step 3, then open it again. |
| No **Open Anyway** button in Privacy & Security | Try opening the app again first (the button appears for about an hour after a blocked attempt), then recheck. Or use the Terminal command. |
| *"ModelVisio" can't be opened because it is from an unidentified developer* | Control-click the app → **Open** → **Open** (macOS 14 and earlier), or the Terminal command. |
| The download link shows *Not Found* | No published release includes the macOS build yet. Check the [Releases page](https://github.com/Premchand006/ModelVisio/releases). |
| The `.dmg` won't open (*"no mountable file systems"* or similar) | The download was interrupted. Delete it and download again. |
| Settings or the API key don't stick, or updates fail | You're probably running it from the disk image. Quit, drag it to **Applications** (step 2), and open it from there. |
| ⌘V doesn't paste into the API key field | You have an old build from before Mac menus were added. Update to the latest version. |
| The AI Copilot says no API key is set | Click the key icon in the AI Copilot panel and paste your Gemini key (step 4). |

## 7. Uninstall

1. Quit ModelVisio (⌘Q).
2. Drag **ModelVisio** from **Applications** to the Trash.
3. *(Optional)* Remove its saved settings, caches, and your stored Gemini API key. Paste this
   into Terminal:

   ```bash
   rm -rf ~/Library/Application\ Support/com.modelvisio.app \
          ~/Library/Caches/com.modelvisio.app \
          ~/Library/WebKit/com.modelvisio.app \
          ~/Library/Saved\ Application\ State/com.modelvisio.app.savedState
   ```

   Some of these folders may not exist; that's fine.

---

**Why the security step?** Apple only skips it for apps signed with a paid Apple Developer ID and
notarized by Apple. ModelVisio is open source, and every installer is built by GitHub Actions
straight from the public source code. Once it's notarized, this step goes away and this guide will be
updated.
