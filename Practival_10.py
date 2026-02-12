# ---------------- HARD COMPUTING ----------------
# Exact rule-based decision (Deterministic)

def hard_temperature(temp):
    if temp < 25:
        return "Cold"
    else:
        return "Hot"


# ---------------- SOFT COMPUTING ----------------
# Approximate decision (Fuzzy-like logic)

def soft_temperature(temp):
    if temp <= 20:
        return "Cold"
    elif 20 < temp <= 30:
        return "Warm"
    else:
        return "Hot"


# ---------------- TEST VALUES ----------------
temperature = [18, 25, 28, 35]

print("Temperature | Hard Computing | Soft Computing")
print("---------------------------------------------")

for t in temperature:
    print(f"{t}°C         | {hard_temperature(t):13} | {soft_temperature(t)}")
