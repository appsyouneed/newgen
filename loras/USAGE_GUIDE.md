# LoRA Management System - Usage Guide

## 🎯 Overview

Your newgen app now has a complete LoRA management system with:
- **JSON-based configuration** with URLs and settings
- **One-click downloads** from Hugging Face/Civitai
- **Automatic trigger prompt injection**
- **Example prompts** dropdown for each LoRA
- **LoRA-specific settings** (weights, recommended steps)

---

## 📥 How to Add New LoRAs

### Method 1: Edit loras.json (Recommended)

1. Open `D:\Apps\newgen\loras\loras.json`
2. Add your LoRA configuration:

```json
{
  "your_lora_name": {
    "display_name": "Your LoRA Display Name",
    "description": "Brief description of what this LoRA does",
    "high_url": "https://huggingface.co/.../high.safetensors",
    "low_url": "https://huggingface.co/.../low.safetensors",
    "high_filename": "your_lora_high.safetensors",
    "low_filename": "your_lora_low.safetensors",
    "trigger_prompt": "special trigger words",
    "prompt_mode": "append",
    "example_prompts": [
      {
        "name": "Example 1",
        "prompt": "Full prompt text here..."
      }
    ],
    "high_weight": 1.0,
    "low_weight": 1.0,
    "recommended_steps": 4,
    "recommended_flow_shift": 6.9,
    "notes": "Any special notes or tips",
    "tags": ["motion", "nsfw"]
  }
}
```

3. Restart the app
4. Click the **📥 Download** button in the UI

### Method 2: Manual Download

1. Download `.safetensors` files manually
2. Place them in `D:\Apps\newgen\loras\`
3. Restart the app
4. They'll appear as "unconfigured" LoRAs

---

## 🎨 Configuration Fields Explained

### Required Fields:
- **`display_name`**: Friendly name shown in UI
- **`high_filename`**: Filename for high-noise LoRA
- **`low_filename`**: Filename for low-noise LoRA

### Download Fields:
- **`high_url`**: Direct download URL for high-noise LoRA
- **`low_url`**: Direct download URL for low-noise LoRA
- Set to `null` if not downloadable or already present

### Prompt Fields:
- **`trigger_prompt`**: Words/phrases that activate the LoRA
  - Set to `null` if no special trigger needed
- **`prompt_mode`**: How to apply trigger prompt
  - `"append"`: Add to end of user's prompt
  - `"prepend"`: Add to beginning of user's prompt
  - `"replace"`: Use example prompts instead (no auto-append)
- **`example_prompts`**: Array of pre-written prompts
  ```json
  "example_prompts": [
    {
      "name": "Simple Motion",
      "prompt": "detailed prompt here..."
    }
  ]
  ```

### Settings Fields:
- **`high_weight`**: LoRA strength for high-noise expert (0.0-2.0, default 1.0)
- **`low_weight`**: LoRA strength for low-noise expert (0.0-2.0, default 1.0)
- **`recommended_steps`**: Optimal inference steps for this LoRA
- **`recommended_flow_shift`**: Optimal flow shift value
- **`resolution_bucket`**: Training resolution (informational)

### Metadata Fields:
- **`description`**: Brief description shown in UI
- **`notes`**: Usage tips, warnings, limitations
- **`tags`**: Array of keywords for organization
- **`auto_enabled`**: Set to `true` to enable by default (usually `false`)

---

## 🚀 Using LoRAs in the App

### Step 1: Download (if needed)
1. Go to **Video Generator** tab
2. Expand **🎨 LoRA Models (Optional)** accordion
3. Click **📥 Download High + Low** button next to LoRA name
4. Wait for download to complete

### Step 2: Enable LoRA
1. Check the checkbox next to the LoRA name
2. LoRA is now active for generation

### Step 3: Use Example Prompts (optional)
1. If LoRA has example prompts, select from dropdown
2. Prompt auto-fills into main Motion & Scene Prompt box
3. Edit as needed

### Step 4: Generate
1. Click **🎬 Generate Video**
2. LoRA is applied automatically
3. Trigger prompt is added if configured

---

## 🔗 Getting Download URLs

### From Hugging Face:
1. Go to model page (e.g., `https://huggingface.co/username/model-name`)
2. Click **Files and versions** tab
3. Find the `.safetensors` file
4. Right-click filename → **Copy link address**
5. URL format: `https://huggingface.co/username/model-name/resolve/main/filename.safetensors`

### From Civitai:
1. Go to model page
2. Click model version you want
3. Click **Download** button → **Copy download link**
4. URL format: `https://civitai.com/api/download/models/{versionId}`
5. For NSFW models, add API key: `?token=YOUR_API_KEY`
   - Get API key from: https://civitai.com/user/account

---

## 📋 Current LoRAs Configured

### 1. Ultimate DeepThroat
- **Status**: Ready to download
- **Files**: High + Low available
- **Mode**: Replace (use example prompts)
- **Examples**: 5 detailed prompts included
- **Notes**: Works best at 4 steps, handles almost every POV

### 2. Ultimate Pussy & Asshole  
- **Status**: Already downloaded
- **Files**: High + Low present
- **Mode**: Append trigger: "HUPODBC, close-up detailed view"
- **Notes**: Best for close-up shots

---

## 🔧 Advanced Usage

### Multiple LoRAs at Once
- Enable multiple checkboxes
- LoRAs stack (may cause conflicts)
- Trigger prompts combine: `base prompt, trigger1, trigger2`

### Manual Prompt Override
- Example prompts are suggestions
- You can edit them or write your own
- Trigger prompt still appends (if mode = append/prepend)

### Weights (Future Feature)
- Currently all LoRAs load at weight 1.0
- Will add sliders for fine-tuning strength

---

## 🐛 Troubleshooting

### LoRA Not Downloading
- Check internet connection
- Verify URL is correct
- For Civitai NSFW: Add API key to URL
- Check console for error messages

### LoRA Not Loading
- Ensure both high AND low files downloaded
- Check file permissions
- Restart app after download
- Verify `.safetensors` format

### Prompt Not Applying
- Check `prompt_mode` in config
- Mode = `replace` means no auto-trigger (use examples instead)
- Mode = `append/prepend` adds trigger automatically

### Example Prompts Not Showing
- Ensure `example_prompts` array exists in config
- Format: `[{"name": "...", "prompt": "..."}]`
- Restart app after editing config

---

## 📝 Tips & Best Practices

1. **Read LoRA descriptions** - Some need specific prompts or settings
2. **Use example prompts** - They're tested and optimized by creators
3. **Start with one LoRA** - Test before combining multiple
4. **Match recommended steps** - LoRA creators know what works best
5. **Backup loras.json** - Before making major changes

---

## 🎉 Example: Adding a New LoRA

Let's add a "Camera Pan" LoRA:

```json
{
  "camera_pan": {
    "display_name": "Smooth Camera Pan",
    "description": "Adds smooth horizontal camera panning motion",
    "high_url": "https://civitai.com/api/download/models/12345",
    "low_url": "https://civitai.com/api/download/models/12346",
    "high_filename": "camera_pan_high.safetensors",
    "low_filename": "camera_pan_low.safetensors",
    "trigger_prompt": "smooth camera pan left to right",
    "prompt_mode": "append",
    "example_prompts": [
      {
        "name": "Slow Pan",
        "prompt": "woman standing, smooth camera pan left to right, slow motion"
      },
      {
        "name": "Fast Pan",
        "prompt": "woman walking, rapid camera pan following movement"
      }
    ],
    "high_weight": 1.0,
    "low_weight": 0.8,
    "recommended_steps": 4,
    "recommended_flow_shift": 5.5,
    "notes": "Best with static subjects. Reduce flow_shift for stronger camera movement.",
    "tags": ["camera", "motion", "pan"]
  }
}
```

Save → Restart → Download → Use! 🚀
