# Plot t-SNE results
fontsize = 11
params = {'axes.labelsize': fontsize + 2,
            'font.size': fontsize,
            'legend.fontsize': fontsize + 1,
            'xtick.labelsize': fontsize,
            'ytick.labelsize': fontsize,
            'pdf.fonttype': 42}
plt.rcParams.update(params)

f, axes = plt.subplots(len(betas), 2, figsize=(6, 3 * len(betas)))
for j, (ax_row, beta) in enumerate(zip(axes, betas)):
    for i, (ax, method, title) in enumerate(zip(ax_row, methods, titles)):
        tsne_data = experiment[method][beta]
        for point, color, marker in zip(tsne_data, colors, markers):
            ax.scatter(point[0], point[1], color=color, s=60,
                        marker=marker, alpha=0.4)
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 0:
            ax.set_ylabel(r"$\beta = %s$" % (np.round(beta, 1)))
        if j == 0:
            ax.set_title(title)
ax.legend(handles=legend_colors, loc=2, ncol=2, bbox_to_anchor=[-0.9, 0.0])

if len(betas) > 2:
    plt.savefig("fig/tsne-meg-all.pdf")
else:
    plt.savefig("fig/tsne-meg.pdf")

plt.show()
