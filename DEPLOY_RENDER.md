# ============================================================
# RENDER.COM FREE TIER DEPLOYMENT GUIDE
# ============================================================

## Prerequisites
1. GitHub repo with your code
2. Model file hosted somewhere (GitHub Releases recommended)
3. Render.com account

---

## STEP 1: Host Your Model File

### Option A: GitHub Releases (Recommended, Free)
1. Go to your GitHub repo → Releases → "Create a new release"
2. Tag: `v1.0` | Title: `Model v1.0`
3. Upload `models/ensemble_final.pkl` as asset
4. Copy the download URL (looks like: `https://github.com/user/repo/releases/download/v1.0/ensemble_final.pkl`)

### Option B: Google Drive (Alternative)
1. Upload to Google Drive
2. Share → "Anyone with link"
3. Convert link: `https://drive.google.com/uc?id=FILE_ID&export=download`

---

## STEP 2: Deploy to Render

### Method 1: Blueprint (Automatic - Recommended)

1. Push all files to GitHub (including `render.yaml`)
2. Go to [dashboard.render.com](https://dashboard.render.com)
3. Click **"New"** → **"Blueprint"**
4. Connect your GitHub repo
4. Render detects `render.yaml` → Click **"Apply"**
5. Add environment variable:
   - Key: `MODEL_URL`
   - Value: Your model download URL from Step 1
6. Click **"Apply"** → Deploy!

### Method 2: Manual Web Service

1. Dashboard → **"New"** → **"Web Service"**
2. Connect GitHub repo
3. Settings:
   - **Name**: `phishshield`
   - **Region**: Oregon (US West) or closest
   - **Branch**: `main`
   - **Runtime**: Python 3
   - **Build Command**: `pip install --no-cache-dir -r requirements.txt`
   - **Start Command**: `gunicorn -b 0.0.0.0:$PORT app:app --timeout 120 --workers 1`
4. **Environment Variables**:
   - `MODEL_URL` = Your model download URL
   - `PYTHON_VERSION` = `3.11.0`
5. Click **"Create Web Service"**

---

## STEP 3: Wait for Deploy

- First deploy: ~5-10 minutes (installs ML deps)
- Subsequent deploys: ~2-3 minutes
- Free tier: Spins down after 15 min inactivity
- First request after sleep: ~30-60s cold start

---

## STEP 4: Verify Deployment

1. Visit: `https://your-app.onrender.com`
2. Test with:
   - `https://google.com` → Should show "Legitimate"
   - `http://paypal-login-security.xyz` → Should show "Phishing"
3. Check health: `https://your-app.onrender.com/health`

---

## IMPORTANT: Free Tier Limitations

| Limitation | Impact |
|------------|--------|
| **Ephemeral filesystem** | Model re-downloads on each deploy |
| **512MB RAM** | Use `--workers 1` |
| **Spins down after 15min inactivity** | ~30-60s cold start |
| **750 hours/month** | ~25 days continuous |
| **No persistent disk** | Model re-downloads each deploy |

---

## MODEL HOSTING OPTIONS (Free)

| Service | Max File | Direct Link | Notes |
|---------|----------|-------------|-------|
| **GitHub Releases** | 2GB | ✅ Yes | Best option |
| **Hugging Face Hub** | Unlimited | ✅ Yes | `hf_hub_download` |
| **Google Drive** | 15GB | ⚠️ Needs conversion | Use `uc?id=ID&export=download` |
| **GitHub LFS** | 2GB free | ✅ Yes | `git lfs track` |
| **SourceForge** | Unlimited | ✅ Yes | Old school but works |

---

## UPLOAD MODEL TO GITHUB RELEASES (One-time)

```bash
# 1. Create release on GitHub web UI
# 2. Or via CLI:
gh release create v1.0 models/ensemble_final.pkl --title "Model v1.0" --notes "Trained ensemble model"

# Get download URL:
# https://github.com/YOUR_USERNAME/phishing-thesis/releases/download/v1.0/ensemble_final.pkl
```

---

## TROUBLESHOOTING

| Error | Solution |
|-------|----------|
| **Build timeout** | Add `--no-cache-dir` to pip, reduce deps |
| **OOM (512MB)** | Use `--workers 1`, reduce model size |
| **Model download fails** | Check URL is direct download, not HTML page |
| **Health check fails** | Ensure `/health` returns 200 in <10s |
| **Module not found** | Check `requirements.txt` has all deps |
| **Cold start too slow** | Accept on free tier, or upgrade |

---

## TEST LOCALLY BEFORE DEPLOY

```bash
# Test with Docker
docker build -t phishshield .
docker run -p 5000:5000 -e MODEL_URL="your_url" phishshield

# Or locally
MODEL_URL=your_url python app.py
```

---

## YOUR LIVE URL WILL BE:

```
https://phishshield.onrender.com
# or
https://your-chosen-name.onrender.com
```

---

## NEXT STEPS AFTER DEPLOY

1. **Custom domain** (Settings → Custom Domain)
2. **Auto-deploys** (enabled by default on push to main)
3. **Monitoring** (Render metrics + `/health` endpoint)
4. **Rate limiting** (add Flask-Limiter if needed)