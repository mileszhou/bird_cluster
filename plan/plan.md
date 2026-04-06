✅ **Script added & version‑controlled**  

- The new batch‑processing script `run_ollama_bird_model.py` is now in the workspace root.  
- It calls an Ollama vision model (e.g., `llava‑phi`) via the local `http://localhost:11434` API, writes a CSV summary (`bird_identification_results.csv`), and updates each XMP file:  

  * adds a generic `bird` keyword for any detected bird,  
  * adds a specific label (e.g., `s‑unknown‑sparrow`) – prefixed with `_` when confidence < 0.80,  
  * marks images with no bird as `_nb`.  

- You can run the script on any machine that has Ollama installed (your more powerful server). Just make sure Python has `requests` and `pillow` installed (`pip install requests pillow`).  

### How to run

```bash
cd /home/robin/.openclaw/workspace
pip install requests pillow   # one‑time
./run_ollama_bird_model.py
```

The script will:

1. Process all JPGs under `data/lr-proj/jpg`.  
2. Update the matching XMP side‑cars in `Photos-2025` and `Photos-2024`.  
3. Produce `bird_identification_results.csv` with columns: `filename, label, confidence, note`.  

After the run you can:

* Search in Lightroom for the underscore (`_`) prefix to find low‑confidence or `_nb` entries.  
* Remove the underscore manually for any correct predictions, or leave them for a later secondary pass (e.g., Merlin).  

Feel free to tweak the `CONF_THRESHOLD` or the Ollama model name inside the script. Let me know when you’ve run it or if you need any adjustments!