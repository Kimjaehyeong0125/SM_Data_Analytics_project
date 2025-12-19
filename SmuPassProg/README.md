# SmuPassProg (minimal)

## Run (VS Code)
1. Open this folder in VS Code.
2. Create/select a Python interpreter.
3. Install deps:
   - `pip install -r requirements.txt`
4. Run:
   - `python main.py --input ./input --competition 10 --trim 0.10`

`--input` should point to a folder that contains the 69 `*_censored.txt` files.
The program will print ranks 1~69, apply 10% two-sided trimming, then mark `pass=True` up to the 10:1 acceptance count.
