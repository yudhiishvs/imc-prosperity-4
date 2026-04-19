import json

log_file = "/Users/vedant/Quant/Prosperity4/imc-prosperity-4/best_strat/307924.log"
with open(log_file) as f:
    data = json.load(f)

activities = data.get("activities", [])
if not activities:
    # try looking at the last few lines or assuming it is just a list of rows
    pass

# Activities is a list of rows with profit loss
last_act = activities[-1] if activities else {}

# find OSMIUM and PEPPER Pnl
osmium_pnl = 0
pepper_pnl = 0

for act in activities[-5:]:
    print(act)

