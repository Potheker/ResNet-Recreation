import matplotlib.pyplot as plt
import history

# Constant data
x = range(181)
c = {3:"olive", 5:"blue", 7:"green", 9:"red", 11:"black"}

# Create subplots
f, (ax1, ax2) = plt.subplots(1,2, figsize=(15,5))

# Plot
for n in (3,5,7,9):
    ax1.plot(x, [y*100 for y in history.read_csv(n,False)[3]], c[n], label=str(6*n+2))
    ax2.plot(x, [y*100 for y in history.read_csv(n,True)[3]], c[n], label=str(6*n+2))
    ax1.plot(x, [y*100 for y in history.read_csv(n,False)[1]], c[n], linestyle="--", linewidth=0.3)
    ax2.plot(x, [y*100 for y in history.read_csv(n,True)[1]], c[n], linestyle="--", linewidth=0.3)

# Design
ax1.set_ylim([0,20])
ax1.axline((0,10),(1,10), linestyle="--", color="grey")
ax1.axline((0,5),(1,5), linestyle="--", color="grey")
ax1.legend(loc="lower left")
ax1.set_xlabel("epoch")
ax1.set_ylabel("error (%), plain")
ax2.set_ylim([0,20])
ax2.axline((0,10),(1,10), linestyle="--", color="grey")
ax2.axline((0,5),(1,5), linestyle="--", color="grey")
ax2.legend(loc="lower left")
ax2.set_xlabel("epoch")
ax2.set_ylabel("error (%), residual")

#
plt.savefig("graphs/graph.png", bbox_inches="tight")
