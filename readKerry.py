import marimo

__generated_with = "0.13.10"
app = marimo.App(width="columns", layout_file="layouts/readKerry.grid.json")


@app.cell(column=0, hide_code=True)
def _():
    # from __future__ import print_function
    import marimo as mo
    import segyio
    import os
    import sys
    import time
    from obspy.io.segy.segy import _read_segy, SEGYBinaryFileHeader
    from obspy import read
    import numpy as np
    from more_itertools import batched
    from functools import reduce
    import matplotlib.pyplot as plt

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Only GPU 1 is visible to this code
    return batched, mo, np, plt, read, reduce, time


@app.cell
def _():
    filename = './Kerry3D.segy'
    return (filename,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Reading the Header Infromations""").center()
    return


@app.cell(hide_code=True)
def _(filename, time):
    t0=time.time()
    segy = _read_segy(filename)
    print('--> data read in {:.1f} sec'.format(time.time()-t0))
    return (segy,)


@app.cell(hide_code=True)
def _(batched, mo, reduce, segy):
    def print_header_informations(segy_instc) -> tuple[int, int, mo.Html, mo.Html]:
        binary_file_header = segy_instc.binary_file_header
        nsample = binary_file_header.number_of_samples_per_data_trace
        # binary_file_header = "| Key | Value |\n\n| --- | --- |\n\n" + '\n'.join(list(
        #     "| " + " | ".join(x.split(":")) + " |\n" 
        #     for x in str(binary_file_header).splitlines() if "unassigned" not in x
        # ))
        _dct = {}
        for _it in str(binary_file_header).splitlines():
            _key, _val = _it.split(":")
            if "unassigned" in _it or _val == "":
                continue
            if int(_val) == 0:
                continue
            _dct[_key.replace("_", " ")] = _val

        binary_file_header = mo.ui.table(_dct, page_size=len(_dct))
        # print("\nbinary_file_header:\n", binary_file_header)

        textual_file_header = str(segy_instc.textual_file_header)[2:-1]
        textual_file_header = list(reduce(lambda i,j: i+j, x) + " "
                                   for x in batched(textual_file_header, 80))
        textual_file_header = "\n".join(textual_file_header)

        data_encoding=segy_instc.data_encoding
        endian=segy_instc.endian
        file=segy_instc.file
        classinfo = segy_instc.__class__
        doc = segy_instc.__doc__
        ntraces=len(segy_instc.traces)
        size_M=segy_instc.traces[0].data.nbytes/1024/1024.*ntraces
        md1 = mo.md(f"""
    ### Textual File Header

    ```raw

    {textual_file_header}

    ```
    """)
        md2 = mo.vstack([
                mo.md(f"""
    ### Binary File Header

    {binary_file_header}

    """),
                mo.hstack([
                    mo.md(f"""
    ### Size of The Data

    - N-Traces: {ntraces}

    - Size in MB: {size_M} MB

    - Size in GB: {size_M/1024} GB
    """),
                    mo.md(f"""
    ### Other Informations

    - Data Encoding: {data_encoding}

    - Data Endianness: {endian}

    - File: {file}

    - ClassInfo: {classinfo}

    - Doc: {doc}

    """)
                ])
            ])
        return ntraces, nsample, md1, md2
    ntraces, nsample, *MD = print_header_informations(segy)
    return MD, nsample, ntraces


@app.cell
def _(MD):
    MD[0]
    return


@app.cell
def _(MD):
    MD[1]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Read use read()""")
    return


@app.cell(hide_code=True)
def _(filename, read, time):
    t0_1 = time.time()
    print('sgy use read:')
    stream = read(filename)
    print('--> data read in {:.1f} min'.format((time.time() - t0_1) / 60))
    return (stream,)


@app.cell(hide_code=True)
def _(stream):
    il = []
    xl = []
    for i_3 in range(len(stream)):
        trace_i_header_3 = stream[i_3].stats.segy.trace_header
        il.append(trace_i_header_3.source_energy_direction_exponent)
        xl.append(trace_i_header_3.ensemble_number)
    return il, xl


@app.cell(hide_code=True)
def _(il, np):
    ilines = np.unique(il)
    n_ilines = len(ilines)
    print("N INLINES:", n_ilines)
    return (n_ilines,)


@app.cell(hide_code=True)
def _(np, xl):
    xlines = np.unique(xl)
    n_xlines = len(xlines)
    print("N CROSSLINES:", n_xlines)
    return (n_xlines,)


@app.cell(hide_code=True)
def _():
    # from collections import Counter
    # t0_2 = time.time()
    # counter = Counter(il)
    # print('Count in {:.1f} sec'.format(time.time() - t0_2))
    # sorted(counter.items())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# this is a cube shape dataset.""")
    return


@app.cell
def _(n_ilines, n_xlines, np, nsample, ntraces, stream, time):
    seis_np = np.zeros((n_ilines, n_xlines, nsample))
    t0_3 = time.time()
    for i_4 in range(ntraces):
        tracei = stream[i_4]
        il_1 = tracei.stats.segy.trace_header.source_energy_direction_exponent
        xl_1 = tracei.stats.segy.trace_header.ensemble_number
        seis_np[il_1 - 510][xl_1 - 58] = tracei.data
    MASK = np.sum(np.abs(seis_np), axis=2)
    MASK = np.where(MASK <= 0.10, True, False)
    for _z in range(nsample):
        seis_np[:, :, _z][MASK] = np.nan

    seis_stats = {}
    seis_flatten_ori = seis_np[~MASK]
    seis_flatten = seis_flatten_ori.flatten()
    SEIS_MAX = np.nanmax(seis_flatten)
    SEIS_MIN = np.nanmin(seis_flatten)
    SEIS_STD = np.nanstd(seis_flatten)
    SEIS_MEAN = np.nanmean(seis_flatten)
    seis_stats['max'] = SEIS_MAX
    seis_stats['min'] = SEIS_MIN
    seis_stats['std'] = SEIS_STD
    seis_stats['mean'] = SEIS_MEAN
    seis_stats['median'] = np.nanmedian(seis_flatten)
    seis_stats = [dict(Statistic=stat, Value=val) for stat, val in seis_stats.items()]

    print('--> data write in {:.1f} min'.format((time.time() - t0_3) / 60))
    return (
        MASK,
        SEIS_MAX,
        SEIS_MEAN,
        SEIS_MIN,
        SEIS_STD,
        seis_flatten,
        seis_flatten_ori,
        seis_np,
        seis_stats,
    )


@app.cell
def _(seis_flatten_ori):
    seis_flatten_ori.shape
    return


@app.cell(hide_code=True)
def _():
    from seiscm import seismic as seiscmap
    return (seiscmap,)


@app.cell
def _(mo, n_ilines, n_xlines):
    inline_full_num = mo.ui.number(start=510,
                                   stop=n_ilines+510,
                                   step=1, label="INLINE NUMBER",
                                   value=143+510)
    xline_full_num = mo.ui.number(start=58,
                                  stop=n_xlines+58,
                                  step=1, label="CROSSLINE NUMBER",
                                  value=100+58,)
    depth_full_num = mo.ui.number(start=0,
                                  stop=5000,
                                  step=4, label="TIME SLICE",
                                  value=120*4)
    return depth_full_num, inline_full_num, xline_full_num


@app.cell
def _(inline_full_num):
    inline_full_num
    return


@app.cell
def _(depth_full_num):
    depth_full_num
    return


@app.cell
def _(xline_full_num):
    xline_full_num
    return


@app.cell(hide_code=True)
def _(
    fig_stat,
    inline_full_num,
    mo,
    n_xlines,
    np,
    nsample,
    plt,
    save_fig_buf,
    seis_np,
    seiscmap,
    vminmax,
):
    plt.figure(figsize=(6,7))
    _idx = inline_full_num.value-510
    # inline 653 plot as: https://wiki.seg.org/wiki/Kerry-3D
    plt.imshow(seis_np[_idx].transpose(), seiscmap(), aspect="auto", **vminmax)
    _n = 50
    plt.ylabel("TWT (s)", fontsize=15)
    plt.xlabel("CROSSLINE", fontsize=15, labelpad=10)
    plt.xticks(
        np.arange(0, n_xlines, _n),
        np.arange(58, n_xlines+58, _n),
        rotation=90)
    plt.yticks(
        np.arange(0, nsample, _n*5),
        np.arange(0, nsample*4, _n*4*5)/1000,
        rotation=0)
    plt.title(f"INLINE Section {inline_full_num.value}", fontsize=20)

    _fig_file_name = ["KerryOriginal", "IL", str(inline_full_num.value)]
    _fig_file_name = "_".join(_fig_file_name)
    _download_lazy = mo.download(
        data = save_fig_buf(f=fig_stat),
        filename = _fig_file_name,
        label = _fig_file_name
    )
    mo.vstack([mo.as_html(plt.gca()).center(), _download_lazy.center()]).center()
    return


@app.cell(hide_code=True)
def _(
    fig_stat,
    mo,
    n_ilines,
    np,
    nsample,
    plt,
    save_fig_buf,
    seis_np,
    seiscmap,
    vminmax,
    xline_full_num,
):
    plt.figure(figsize=(6,7))
    _idx = xline_full_num.value-58
    _n = 50
    plt.imshow(seis_np[:,_idx,:].transpose(), seiscmap(), aspect="auto", **vminmax)
    plt.ylabel("TWT (s)", fontsize=15)
    plt.xlabel("INLINE", labelpad=10, fontsize=15)
    plt.xticks(
        np.arange(0, n_ilines, _n),
        np.arange(510, n_ilines+510, _n),
        rotation=90)
    plt.yticks(
        np.arange(0, nsample, _n*5),
        np.arange(0, nsample*4, _n*4*5)/1000,
        rotation=0)
    plt.title(f"CROSSLINE Section {xline_full_num.value}", fontsize=20)
    _fig_file_name = ["KerryOriginal", "XL", str(xline_full_num.value)]
    _fig_file_name = "_".join(_fig_file_name)
    _download_lazy = mo.download(
        data = save_fig_buf(f=fig_stat),
        filename = _fig_file_name,
        label = _fig_file_name
    )
    mo.vstack([mo.as_html(plt.gca()).center(), _download_lazy.center()]).center()
    return


@app.cell(hide_code=True)
def _(SEIS_MAX, SEIS_MIN):
    vminmax = dict(vmax=SEIS_MAX, vmin=SEIS_MIN)
    return (vminmax,)


@app.cell(hide_code=True)
def _(
    depth_full_num,
    fig_stat,
    inline_full_num,
    mo,
    n_ilines,
    n_xlines,
    np,
    plt,
    save_fig_buf,
    seis_np,
    seiscmap,
    vminmax,
):
    plt.figure(figsize=(10,8))
    _idx = depth_full_num.value // 4
    plt.imshow(seis_np[:,:,_idx], seiscmap(), aspect="equal", origin='upper', **vminmax)
    _n = 50
    plt.ylabel("INLINE", fontsize=15)
    plt.xlabel("CROSSLINE", fontsize=15)
    plt.yticks(
        np.arange(0, n_ilines, _n),
        np.arange(58, n_ilines+58, _n),
        rotation=90)
    plt.xticks(
        np.arange(0, n_xlines, _n),
        np.arange(510, n_xlines+510, _n),
        rotation=0)
    plt.title(f"TIME SLICE {depth_full_num.value} ms")
    # mo.vstack([
    #     mo.md(f"## $\\texttt{{TIME SLICE}}$ {depth_full_num.value} ms").center(),
    #     mo.as_html(plt.gca()) ])

    _fig_file_name = ["KerryOriginal", "Z", str(inline_full_num.value)]
    _fig_file_name = "_".join(_fig_file_name)
    _download_lazy = mo.download(
        data = save_fig_buf(f=fig_stat),
        filename = _fig_file_name,
        label = _fig_file_name
    )
    mo.vstack([mo.as_html(plt.gca()).center(), _download_lazy.center()]).center()
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(MASK, mo, np, seis_np, time):
    def save_on_click(data=seis_np, suffix="", save_mask=False):
        t0_4 = time.time()
        with mo.status.progress_bar(
            total=2,
            title="Saving The Data",
            subtitle="Please wait",
        ) as _bar:
            np.savez('./kerry3d' + suffix, data)
            if save_mask:
                np.savez('./kerry3d_mask', MASK.astype(np.int16))
            _bar.update()
        return mo.md(
            '## `segy` Saved to npz:\ndata save in {:.1f} min'.format(
                (time.time() - t0_4) / 60)
        ).center()

    save_value = 0
    run_button = mo.ui.run_button(label="Save `.npz`")
    run_button.center()
    return run_button, save_on_click


@app.cell
def _(mo, run_button, save_on_click):
    if run_button.value:
        _a = save_on_click(save_mask=True)
    else:
        _a = mo.md("### click button to save!").center()
    _a
    return


@app.cell
def _():
    return


@app.cell(column=1)
def _():
    from io import BytesIO, StringIO
    import pandas as pd

    return BytesIO, pd


@app.cell
def _(mo):
    mo.md(r"""# Data Preprocessing """).center()
    return


@app.cell
def _(BytesIO):
    from matplotlib import colors
    import matplotlib.patches as pltPatches
    import matplotlib.path as pltPath
    from itertools import cycle

    def save_fig_buf(f):
            buf = BytesIO()
            f.savefig(buf, format="png")
            return buf

    def save_tex_buf(string):
        return string.encode("utf-8")
    return cycle, pltPatches, pltPath, save_fig_buf, save_tex_buf


@app.cell(hide_code=True)
def _(mo):
    hist_binsize = mo.ui.number(start=10, stop=500, step=5, value=100, label="Histogram BinSize")

    return (hist_binsize,)


@app.cell
def _(mo):
    mo.md(r"""## Original""").center()
    return


@app.cell(hide_code=True)
def _(mo, pd, save_tex_buf, seis_stats):
    _df = pd.DataFrame(seis_stats)
    _filename = "KerryOriginal_hist"
    _latex_tabel = f"""\\begin{{tabel}}[ht!]
    \\caption{{{_filename}}}
    \\label{{tab:{_filename}}}
    {_df.to_latex(index=False)}\\end{{tabel}}"""
    _download_lazy = mo.download(
        data = save_tex_buf(_latex_tabel),
        filename = _filename + ".tex",
        label=_filename + ".tex"
    )
    mo.vstack([
        _download_lazy.center(),
        mo.ui.text_area(_latex_tabel, full_width=True, max_length=100),
    ])
    return


@app.cell(hide_code=True)
def _(mo, seis_stats):
    table_stat = mo.ui.table(data=seis_stats, pagination=False)
    table_stat
    return (table_stat,)


@app.cell(hide_code=True)
def _(hist_binsize):
    hist_binsize.center()
    return


@app.cell(hide_code=True)
def _(plt):
    fig_stat, ax_stat = plt.subplots(1,1, figsize=(7,5))
    return ax_stat, fig_stat


@app.cell(hide_code=True)
def _(hist_binsize, np, pltPatches, pltPath, seis_flatten):
    def make_histogram_seis(data=seis_flatten, in_bins=hist_binsize.value):
        n, bins = np.histogram(data, in_bins)

        # get the corners of the rectangles for the histogram
        left = bins[:-1]; right = bins[1:]
        bottom = np.zeros(len(left))
        top = bottom + n

        # we need a (numrects x numsides x 2) numpy array for the path helper
        # function to build a compound path
        XY = np.array([[left, left, right, right], [bottom, top, top, bottom]]).T

        # get the Path object
        barpath = pltPath.Path.make_compound_path_from_polys(XY)
        patch = pltPatches.PathPatch(barpath,
                                     edgecolor=None,
                                     facecolor="gray", alpha=0.5, linewidth=0.1)
        patch.sticky_edges.y[:] = [0]
        return n, bins, patch
    N, bins, patches = make_histogram_seis()
    return make_histogram_seis, patches


@app.cell(hide_code=True)
def _(
    ax_stat,
    cycle,
    fig_stat,
    hist_binsize,
    mo,
    patches,
    save_fig_buf,
    table_stat,
):
    line_colors_stats = cycle(["r", 'k', 'b', 'g', 'orange'])

    n_stats_to_plot = len(table_stat.value)
    ax_stat.clear(); ax_stat.add_patch(patches)
    _fig_file_name = ["KerryOriginal", "hist", str(hist_binsize.value)]

    if n_stats_to_plot != 0:
        for _s, _c in zip(table_stat.value, line_colors_stats):
            ax_stat.axvline(_s["Value"],
                            color=_c, linestyle="dashed",
                            linewidth=1,
                            label=f"${_s['Statistic'].title()}={_s['Value']:.3f}$")
            _fig_file_name += [_s['Statistic']]
        ax_stat.legend(ncols=n_stats_to_plot//2, bbox_to_anchor =(0.5,-0.37), loc='lower center')
        _title = "Histogram of Seismic Amplitude"
        ax_stat.set_ylabel("Frequency", fontstyle="italic")
        ax_stat.set_xlabel("Amplitude", fontstyle="italic")
    else:
        _title = "Histogram of Seismic' Amplitude and Pick Statistic Values to plot, btw!"

    ax_stat.grid(which="both",alpha=0.25)
    ax_stat.autoscale_view()
    # fig_stat.suptitle(_title, fontsize=18, fontstyle="italic")
    _fig_file_name = "_".join(_fig_file_name)
    _download_lazy = mo.download(
        data = save_fig_buf(f=fig_stat),
        filename = _fig_file_name,
        label = _fig_file_name
    )
    mo.vstack([mo.as_html(fig_stat).center(), _download_lazy.center()]).center()
    # plt.show()        
    return (line_colors_stats,)


@app.cell
def _(mo):
    mo.md(r"""## Standardization""").center()
    return


@app.cell
def _(SEIS_MEAN, SEIS_STD, np, seis_flatten_ori):
    seis_flatten_stdr_ori = (seis_flatten_ori - SEIS_MEAN)/(SEIS_STD)
    seis_flatten_stdr = seis_flatten_stdr_ori.flatten()
    SEIS_MAX_stdr = np.nanmax(seis_flatten_stdr)
    SEIS_MIN_stdr = np.nanmin(seis_flatten_stdr)
    seis_stats_stdr = {}
    seis_stats_stdr['max'] = SEIS_MAX_stdr
    seis_stats_stdr['min'] = SEIS_MIN_stdr
    seis_stats_stdr['std'] = np.nanstd(seis_flatten_stdr)
    seis_stats_stdr['mean'] = np.nanmean(seis_flatten_stdr)
    seis_stats_stdr['median'] = np.nanmedian(seis_flatten_stdr)
    seis_stats_stdr = [dict(Statistic=stat, Value=val) for stat, val in seis_stats_stdr.items()]
    return seis_flatten_stdr, seis_flatten_stdr_ori, seis_stats_stdr


@app.cell(hide_code=True)
def _(mo, pd, save_tex_buf, seis_stats_stdr):
    _df = pd.DataFrame(seis_stats_stdr)
    _filename = "KerryStandardize_hist"
    _latex_tabel = f"""\\begin{{tabel}}[ht!]
    \\caption{{{_filename}}}
    \\label{{tab:{_filename}}}
    {_df.to_latex(index=False)}\\end{{tabel}}"""
    _download_lazy = mo.download(
        data = save_tex_buf(_latex_tabel),
        filename = _filename + ".tex",
        label=_filename + ".tex"
    )
    mo.vstack([
        _download_lazy.center(),
        mo.ui.text_area(_latex_tabel, full_width=True, max_length=100),
    ])
    return


@app.cell
def _(mo, seis_stats_stdr):
    table_stat_stdr = mo.ui.table(data=seis_stats_stdr, pagination=False)
    table_stat_stdr
    return (table_stat_stdr,)


@app.cell
def _(plt):
    fig_stat_stdr, ax_stat_stdr = plt.subplots(1,1, figsize=(7,5))
    return ax_stat_stdr, fig_stat_stdr


@app.cell
def _(make_histogram_seis, seis_flatten_stdr):
    N_stdr, bins_stdr, patches_stdr = make_histogram_seis(data=seis_flatten_stdr)
    return (patches_stdr,)


@app.cell(hide_code=True)
def _(
    ax_stat_stdr,
    fig_stat_stdr,
    hist_binsize,
    line_colors_stats,
    mo,
    patches_stdr,
    save_fig_buf,
    table_stat_stdr,
):
    n_stats_to_plot_stdr = len(table_stat_stdr.value)
    ax_stat_stdr.clear();
    ax_stat_stdr.add_patch(patches_stdr)
    _fig_file_name = ["Kerrystandardized", "hist", str(hist_binsize.value)]

    if n_stats_to_plot_stdr != 0:
        for _s, _c in zip(table_stat_stdr.value, line_colors_stats):
            ax_stat_stdr.axvline(_s["Value"],
                            color=_c, linestyle="dashed",
                            linewidth=1,
                            label=f"${_s['Statistic'].title()}={_s['Value']:.3f}$")
            _fig_file_name += [_s['Statistic']]
        ax_stat_stdr.legend(ncols=n_stats_to_plot_stdr//2, bbox_to_anchor =(0.5,-0.37), loc='lower center')
        _title = "Histogram of Seismic Amplitude"
        ax_stat_stdr.set_ylabel("Frequency", fontstyle="italic")
        ax_stat_stdr.set_xlabel("Amplitude", fontstyle="italic")
    else:
        _title = "Histogram of Seismic' Amplitude and Pick Statistic Values to plot, btw!"

    ax_stat_stdr.grid(which="both",alpha=0.25)
    ax_stat_stdr.autoscale_view()
    # fig_stat_stdr.suptitle(_title, fontsize=18, fontstyle="italic")
    _fig_file_name = "_".join(_fig_file_name)
    _download_lazy = mo.download(
        data = save_fig_buf(f=fig_stat_stdr),
        filename = _fig_file_name,
        label = _fig_file_name
    )
    mo.vstack([mo.as_html(fig_stat_stdr).center(), _download_lazy.center()]).center()
    # plt.show()        
    return


@app.cell
def _(MASK, mo, n_ilines, n_xlines, np, nsample, seis_flatten_stdr_ori):
    _data = np.zeros((n_ilines, n_xlines, nsample))
    _data[~MASK] = seis_flatten_stdr_ori
    run_button_stdr = mo.ui.run_button(
        label="Save `.npz` standardize")
    run_button_stdr.center()
    return (run_button_stdr,)


@app.cell
def _(mo, run_button_stdr, save_on_click):
    if run_button_stdr.value:
        _a = save_on_click(save_mask=True, suffix="_stdr")
    else:
        _a = mo.md("### click button to save!").center()
    _a
    return


if __name__ == "__main__":
    app.run()
