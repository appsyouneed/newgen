# LoRA Auto-Settings & Compatibility System

## 🎯 Overview

The app now automatically applies LoRA-recommended settings while allowing you to override them. It also checks compatibility when using multiple LoRAs simultaneously.

---

## ⚙️ **How Auto-Settings Work:**

### **1. Single LoRA Selected**

When you enable ONE LoRA:
- ✅ **Steps** auto-set to LoRA's `recommended_steps`
- ✅ **Flow Shift** auto-set to LoRA's `recommended_flow_shift`
- ✅ **Weights** applied automatically (high/low)
- ✅ **Trigger prompt** auto-appended (if configured)

**Example:**
```
You enable: "Cowgirl & Reverse Cowgirl"
Auto-applied:
  - Steps: 4 (from LoRA config)
  - Flow Shift: 6.9 (from LoRA config)
  - Weights: High=0.8, Low=0.8
  - Trigger: "r3v3rs3_c0wg1rl" added to your prompt
```

---

### **2. Multiple LoRAs Selected**

When you enable MULTIPLE LoRAs:

#### **Compatible LoRAs** (same settings):
- ✅ Settings match perfectly
- ✅ Shared settings applied
- ✅ Weights averaged
- ✅ All triggers combined

**Example:**
```
You enable:
  - "Sex Smash Cut" (steps=4, flow=6.9)
  - "Cowgirl & Reverse Cowgirl" (steps=4, flow=6.9)

Result: ✓ Compatible!
  - Steps: 4 (both agree)
  - Flow Shift: 6.9 (both agree)
  - Weights: High=0.8, Low=0.8 (averaged)
  - Triggers: Both added to prompt
```

#### **Incompatible LoRAs** (conflicting settings):
- ⚠️ Settings conflict detected
- ⚠️ Warning displayed in UI
- 🎛️ Defaults used (you can override)
- ✅ Both still work, just not optimized

**Example:**
```
You enable:
  - "Ultimate Pussy & Anus Helper" (steps=10, flow=6.9)
  - "Sex Smash Cut" (steps=4, flow=6.9)

Result: ⚠️ Conflicting!
  - Warning: "Step conflict: 10, 4 steps recommended"
  - Uses: Default steps (3) unless you override
  - Flow: 6.9 (both agree)
  - Both LoRAs still load and work
```

---

## 🎛️ **User Override System**

### **How to Override:**

1. **Go to Advanced Settings accordion**
2. **Change the slider** (Steps, Flow Shift, etc.)
3. **Your value takes priority** over LoRA recommendations

### **Override Detection:**

The app knows when you've overridden:
```
🎛️ Using your steps: 6 (override active, LoRA recommends 4)
```

vs automatic:
```
📊 Auto-set steps: 4 (LoRA recommendation)
```

### **When Auto-Settings Apply:**

✅ **Settings ARE auto-applied when:**
- Value is at DEFAULT (3 steps, 6.9 flow shift)
- Flow Shift Auto mode is OFF
- LoRA has `recommended_steps` or `recommended_flow_shift`

❌ **Settings NOT auto-applied when:**
- You've manually changed the slider
- Flow Shift Auto mode is ON (duration-based override)
- LoRA has no recommendations

---

## 📊 **Compatibility Status Display**

### **Real-Time Feedback:**

When you check LoRA checkboxes, a status box appears:

```
### 🎨 Active LoRAs: 2

✓ Compatible: Sex Smash Cut, Cowgirl & Reverse Cowgirl
Recommended Steps: 4
Recommended Flow Shift: 6.9
Average Weights: High=0.80, Low=0.80
```

OR

```
### 🎨 Active LoRAs: 2

⚠️ Combining: Ultimate Pussy & Anus Helper, Sex Smash Cut
⚠️ Step conflict: 10, 4 steps recommended
Recommended Flow Shift: 6.9
Average Weights: High=0.78, Low=0.88

⚠️ Note: Settings conflict - using defaults. You can manually override in Advanced Settings.
```

---

## 🔧 **Technical Details**

### **Settings Priority (High to Low):**

1. **User Manual Override** - You changed the slider
2. **LoRA Recommendation** - From loras.json config
3. **App Default** - WAN_STEPS=3, WAN_FLOW_SHIFT=6.9

### **Weight Averaging Formula:**

When combining multiple LoRAs:
```python
merged_high_weight = sum(all_high_weights) / count
merged_low_weight = sum(all_low_weights) / count
```

**Example:**
- LoRA A: high=1.0, low=0.95
- LoRA B: high=0.8, low=0.8
- **Result:** high=0.90, low=0.875

### **Compatibility Check Logic:**

```python
Compatible IF:
  - All recommended_steps are the same (or empty)
  - All recommended_flow_shift are the same (or empty)

Incompatible IF:
  - LoRA A wants 10 steps, LoRA B wants 4 steps
  - LoRA A wants flow=5.0, LoRA B wants flow=6.9
```

---

## 💡 **Best Practices**

### **For Single LoRAs:**
1. ✅ Let auto-settings apply (don't touch sliders)
2. ✅ Use example prompts for best results
3. ✅ Check the notes for special requirements

### **For Multiple LoRAs:**
1. 🔍 Check compatibility status display
2. ⚠️ If conflicting, manually set steps/flow shift
3. 🧪 Test combinations before long generations
4. 📝 Save working combinations

### **Compatible Combinations:**
- **Sex Smash Cut + Cowgirl** ✓ (both 4 steps, 6.9 flow)
- **Ultimate DeepThroat + Combo Handjob** ✓ (both 4 steps, 6.9 flow)
- **Any LoRAs with same settings** ✓

### **Incompatible Combinations:**
- **Ultimate Pussy/Anus (10 steps) + Sex Smash Cut (4 steps)** ⚠️
- Manually set to 6-8 steps as compromise

---

## 🎯 **Console Output Examples**

### **Single LoRA (Auto-Applied):**
```
✓ Using Cowgirl & Reverse Cowgirl
📊 Auto-set steps: 4 (LoRA recommendation)
📊 Auto-set flow_shift: 6.9 (LoRA recommendation)
Applying 1 LoRA(s): ['cowgirl_reverse_cowgirl']
```

### **Multiple Compatible:**
```
✓ Compatible: Sex Smash Cut, Cowgirl & Reverse Cowgirl
📊 Auto-set steps: 4 (LoRA recommendation)
📊 Auto-set flow_shift: 6.9 (LoRA recommendation)
Applying 2 LoRA(s): ['sex_smash_cut', 'cowgirl_reverse_cowgirl']
```

### **Multiple Incompatible:**
```
⚠️ Combining: Ultimate Pussy & Anus Helper, Sex Smash Cut
⚠️ Step conflict: 10, 4 steps recommended
🎛️ Using your steps: 6 (override active, LoRA recommends varying)
Applying 2 LoRA(s): ['ulitmate_pussy_asshole', 'sex_smash_cut']
```

### **User Override:**
```
✓ Using Sex Smash Cut
🎛️ Using your steps: 8 (override active, LoRA recommends 4)
📊 Auto-set flow_shift: 6.9 (LoRA recommendation)
Applying 1 LoRA(s): ['sex_smash_cut']
```

---

## ❓ **FAQ**

### **Q: Can I use 3+ LoRAs at once?**
A: Yes! The system checks compatibility for any number. More LoRAs = higher chance of conflicts.

### **Q: What if I want to ignore recommendations?**
A: Just change the slider in Advanced Settings. Your value always takes priority.

### **Q: Do weights get applied automatically?**
A: Yes, weights are always applied per LoRA config. Future: weight sliders for manual control.

### **Q: Can I save my override settings?**
A: Not yet. Future feature: preset combinations with custom settings.

### **Q: What if LoRA has no recommended settings?**
A: Defaults are used. Example prompts and triggers still work.

### **Q: Does this work with custom LoRAs?**
A: Yes! Add them to loras.json with recommended_steps and recommended_flow_shift fields.

---

## 🚀 **Summary**

✅ **Single LoRA:** Settings auto-apply, you can override  
✅ **Compatible Multi:** Settings auto-apply, weights averaged  
⚠️ **Incompatible Multi:** Warnings shown, manual override recommended  
🎛️ **User Override:** Always respected, indicated in console  
📊 **Real-Time Feedback:** Status display updates as you check boxes  

**The system is smart enough to help, but always respects your control!** 🎉
