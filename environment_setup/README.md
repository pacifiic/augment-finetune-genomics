## 🧩 Environment Setup

To set up your environment, follow the steps below.

---

### 1. Create the Conda Environment

```bash
conda env create -f /tmp/pytorch_enformer.yml -n pytorch_enformer
conda activate pytorch_enformer
```

This command creates a new Conda environment named **`pytorch_enformer`** using the specified YAML file.

---

### 2. Install Required Packages

```bash
pip install -r ./pip_requirements.txt
```

> ⚠️ Note: During this step, some dependencies may fail to install due to version conflicts or compatibility issues between packages.  
> If that happens, continue with the next step and install the necessary packages individually.

---

### 3. Manually Install Key Packages

After completing the above steps, manually install the following packages to ensure proper compatibility:

```bash
pip install deepspeed==0.15.1
pip install pandas==2.2.2
pip install scipy==1.13.1
pip install peft==0.14.0
pip install enformer-pytorch==0.8.8
pip install transformers==4.45.2
```

---

### ✅ Summary

These steps ensure your environment is correctly configured for running **Enformer-based fine-tuning with LoRA and DeepSpeed**.  
If installation errors persist, try upgrading `pip` and re-running the commands:

```bash
pip install --upgrade pip
```
