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
    from pathlib import Path

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Only GPU 1 is visible to this code
    return Path, batched, mo, np, plt, read, reduce, time


@app.cell(hide_code=True)
def _(Path):
    filedir = Path(__file__).parent
    filename = filedir / "Kerry3D.segy"
    kerry_url = "http://s3.amazonaws.com/open.source.geoscience/open_data/newzealand/Taranaiki_Basin/Keri_3D/Kerry3D.segy"
    return filedir, filename, kerry_url


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# 3D Seismic Kerry Data Gathering & Preprocessing""").center()
    return


@app.cell(hide_code=True)
def _(filename, kerry_url, mo, read):
    if filename.exists():
        _ntask = 2
        _spin = mo.status.progress_bar(
            title="Preparing the SEGY Data ...",
            completion_title="SEGY Data is ready",
            total=_ntask)
        with _spin as _spinner:
            _spinner.update(subtitle="Loading the data")
            segy = _read_segy(filename)
            _spinner.update(subtitle="Reading data as stream")
            stream = read(filename)
    else:
        _ntask = 3
        import subprocess
        _spin = mo.status.progress_bar(
            title="Preparing the SEGY Data ...",
            completion_title="SEGY Data is ready",
            total=_ntask)
        with _spin as _spinner:
            _spinner.update(
                subtitle=f"Downloading from {kerry_url} to {str(filename)}")
            subprocess.run([
                "wget",
                "-O",
                "./Kerry3D/Kerry3D.segy",
                kerry_url])
            _spinner.update(subtitle="Loading the data")
            segy = _read_segy(filename)
            _spinner.update(subtitle="Reading data as stream")
            stream = read(filename)

    return segy, stream


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
def _(mo, np, stream):
    # collection data from stream
    n_stream = len(stream)+1
    Bar_collecting_il_xl = mo.status.progress_bar(
        title="Collectin INLINE & CROSSLINE Indexes",
        completion_title="INLINE & CROSSLINE Indexes are collected",
        total = n_stream
    )

    il = []
    xl = []
    n_ilines = 0
    n_xlines = 0
    with Bar_collecting_il_xl as _bar:
        for i_3 in range(n_stream-1):
            _bar.update(subtitle=f"Collecting from stream-{i_3}")
            trace_i_header_3 = stream[i_3].stats.segy.trace_header
            il.append(trace_i_header_3.source_energy_direction_exponent)
            xl.append(trace_i_header_3.ensemble_number)

        ilines = np.unique(il)
        n_ilines = len(ilines)
        xlines = np.unique(xl)
        n_xlines = len(xlines)
        _bar.update(subtitle=f"Got {n_ilines} INLINES & {n_xlines} CROSSLINES")
    return n_ilines, n_xlines


@app.cell(hide_code=True)
def _(mo, n_ilines, n_xlines, np, nsample, ntraces, stream):
    # streaming traces
    Bar_collecting_trace = mo.status.progress_bar(
        title="Collectin Data from Traces",
        completion_title="Data are collected & masked",
        total = ntraces + nsample
    )

    seis_np = np.zeros((n_ilines, n_xlines, nsample))

    with Bar_collecting_trace as _bar:
        for i_4 in range(ntraces):
            _bar.update(subtitle=f"Colleting data from trace-{i_4}")
            tracei = stream[i_4]
            il_1 = tracei.stats.segy.trace_header.source_energy_direction_exponent
            xl_1 = tracei.stats.segy.trace_header.ensemble_number
            seis_np[il_1 - 510][xl_1 - 58] = tracei.data

        MASK = np.sum(np.abs(seis_np), axis=2)
        MASK = np.where(MASK == 0.000, True, False)
        for _z in range(nsample):
            _bar.update(subtitle=f"Masking data at vertical silce-{_z}")
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
    return (
        MASK,
        SEIS_MAX,
        SEIS_MEAN,
        SEIS_MIN,
        SEIS_STD,
        seis_flatten,
        seis_np,
        seis_stats,
    )


@app.cell(hide_code=True)
def _():
    from seiscm import seismic as seiscmap
    from io import BytesIO, StringIO
    import pandas as pd
    from matplotlib import colors
    import matplotlib.patches as pltPatches
    import matplotlib.path as pltPath
    from itertools import cycle
    return BytesIO, cycle, pd, pltPatches, pltPath, seiscmap


@app.cell
def _(BytesIO, plt):
    def save_fig_buf(f):
            buf = BytesIO()
            plt.gcf()
            plt.savefig(buf, bbox_inches='tight', format="png")
            return buf

    def save_tex_buf(string):
        return string.encode("utf-8")
    return save_fig_buf, save_tex_buf


@app.cell(hide_code=True)
def _(n_ilines, n_xlines, nsample):
    sample_rate = 4
    IL_START = 58
    XL_START = 510
    Z_START  = 0 * sample_rate

    IL_END = n_ilines + IL_START
    XL_END = n_xlines + XL_START
    Z_END  = nsample  * sample_rate
    return IL_END, IL_START, XL_END, XL_START, Z_END, Z_START, sample_rate


@app.cell(hide_code=True)
def _():
    from collections import namedtuple

    TpVal = namedtuple("TpVal", ['value'])
    return (TpVal,)


@app.cell
def _(IL_END, IL_START, XL_END, XL_START, Z_END, Z_START, mo, sample_rate):
    boundary_form = mo.md("""
    | Boundary  | Start | Stop  |
    | :-------: | :---: | :--:  |
    | CROSSLINE | {xl1} | {xl2} |
    | INLINE    | {il1} | {il2} |
    | DEPTH     | {z1}  | {z2}  |
    """).center().batch(
        xl1 = mo.ui.number(start=XL_START, stop=XL_END, step=1, label="", value=45 + 510),
        xl2 = mo.ui.number(start=XL_START, stop=XL_END, step=1, label="", value=45 + 510 + 608),
        il1 = mo.ui.number(start=IL_START, stop=IL_END, step=1, label="", value=73,),
        il2 = mo.ui.number(start=IL_START, stop=IL_END, step=1, label="", value=73 + 192,),
        z1 = mo.ui.number(start=Z_START,  stop=Z_END,  step=sample_rate, label="", value=25*4),
        z2 = mo.ui.number(start=Z_START,  stop=Z_END,  step=sample_rate, label="", value=(272 + 25) * 4),
    ).form()
    boundary_form
    return (boundary_form,)


@app.cell
def _(boundary_form):
    boundary_form.value
    return


@app.cell(hide_code=True)
def _(TpVal, boundary_form):
    xline_full_num  = TpVal(boundary_form.value["xl1"]) 
    inline_full_num = TpVal(boundary_form.value["il1"]) 
    depth_full_num  = TpVal(boundary_form.value["z1"]) 

    #mo.ui.number(start=XL_START, stop=XL_END, step=1, label="CROSSLINE NUMBER", value=610)
    #mo.ui.number(start=IL_START, stop=IL_END, step=1, label="INLINE NUMBER", value=76,)
    #mo.ui.number(start=Z_START,  stop=Z_END,  step=sample_rate, label="TIME SLICE", value=120*4)

    xline_full_num_2 = TpVal(boundary_form.value["xl2"])
    inline_full_num_2 = TpVal(boundary_form.value["il2"])
    depth_full_num_2 = TpVal(boundary_form.value["z2"])

    # xline_full_num_2 = mo.ui.number(start=XL_START, stop=XL_END, step=1, label="CROSSLINE NUMBER", value=1200)
    # inline_full_num_2 = mo.ui.number(start=IL_START, stop=IL_END, step=1, label="INLINE NUMBER", value=270,)
    # depth_full_num_2 = mo.ui.number(start=Z_START,  stop=Z_END,  step=sample_rate, label="TIME SLICE", value=2000)
    return (
        depth_full_num,
        depth_full_num_2,
        inline_full_num,
        inline_full_num_2,
        xline_full_num,
        xline_full_num_2,
    )


@app.cell(hide_code=True)
def _(
    IL_START,
    SEIS_MAX,
    SEIS_MIN,
    XL_START,
    Z_START,
    depth_full_num,
    depth_full_num_2,
    inline_full_num,
    inline_full_num_2,
    xline_full_num,
    xline_full_num_2,
):
    vminmax = dict(vmax=SEIS_MAX, vmin=SEIS_MIN)

    il_idx = inline_full_num.value-IL_START
    xl_idx = xline_full_num.value-XL_START
    z_idx = (depth_full_num.value - Z_START) // 4

    il_idx_2 = inline_full_num_2.value-IL_START
    xl_idx_2 = xline_full_num_2.value-XL_START
    z_idx_2 = (depth_full_num_2.value - Z_START) // 4
    return il_idx, il_idx_2, vminmax, xl_idx, xl_idx_2, z_idx, z_idx_2


@app.cell(hide_code=True)
def _(mo, np, plt, save_fig_buf, seiscmap, vminmax):
    def plot_ilines(
        data        : np.ndarray,
        idxs        : tuple[int, int],
        buttons     : tuple[int, int],
        vertbuttons : tuple[int, int],
        horbuttons  : tuple[int, int],
        vertidxs    : tuple[int, int],
        horidxs     : tuple[int, int],
        dims        : tuple[int, int, int],
        dim_start   : tuple[int, int, int],
        title       : str = "KerryOriginal",
        minmax      : dict = vminmax,
    ) -> mo.Html:
        fig_iline, ax_iline = plt.subplots(1,2, figsize=(10,7), sharey=False, layout="compressed")
        ax_iline[0].imshow(data[idxs[0], :, :].transpose(), seiscmap(), aspect="auto", **minmax)
        ax_iline[1].imshow(data[idxs[1], :, :].transpose(), seiscmap(), aspect="auto", **minmax)
        _n = 75
        for _ax, _d in zip(ax_iline, buttons):
            _ax.set_ylabel("TWT (s)", fontsize=15)
            _ax.set_xlabel("CROSSLINE", fontsize=15, labelpad=10)

            _ax.set_xticks(
                np.arange(0, dims[1], _n),
                np.arange(dim_start[1], dims[1]+dim_start[1], _n),
                rotation=90)
            _ax.set_yticks(
                np.arange(0, dims[2], _n),
                np.arange(0, (dims[2])*4, _n*4)/1000,
                rotation=0)

            # _axT = _ax.secondary_xaxis('top')
            # _axT.set_xticks(
            #     np.arange(0, dims[1], _n),
            #     np.arange(dim_start[1], dims[1]+dim_start[1], _n),
            #     rotation=90)
        
            _axR = _ax.secondary_yaxis('right')
            _axR.set_yticks(
                np.arange(0, dims[2], _n),
                np.arange(0, (dims[2])*4, _n*4)/1000,
                rotation=0)

            _ax.set_title(f"$INLINE\ {_d}$", style="italic", fontsize=20, pad=10)
            _ax.axvline(vertidxs[0], color="blue", label=f"CROSSLINE {vertbuttons[0]}")
            _ax.axvline(vertidxs[1], color="blue", linestyle="--", label=f"CROSSLINE {vertbuttons[1]}")
            _ax.axhline(horidxs[0], color="black", label=f"TWT {horbuttons[0]} ms")
            _ax.axhline(horidxs[1], color="black", linestyle="--", label=f"TWT {horbuttons[1]} ms")
            _ax.grid(which="both", color="black", alpha=0.25, markevery=2, snap=True)

        _handles, _labels = ax_iline[1].get_legend_handles_labels()
        fig_iline.legend(_handles, _labels, ncols=4, loc='lower center', bbox_to_anchor =(0.5,-0.075))

        _fig_file_name = [title,
                          "IL", str(buttons[0])     + "-" + str(buttons[1]),
                          "XL", str(vertbuttons[0]) + "-" + str(vertbuttons[1]),
                          "TWT", str(horbuttons[0]) + "-" + str(horbuttons[1])
                         ]
        _fig_file_name = "_".join(_fig_file_name)
        _download_lazy = mo.download(
            data = save_fig_buf(plt.gcf()),
            filename = _fig_file_name,
            label = _fig_file_name
        )
        return mo.vstack([mo.as_html(plt.gcf()).center(), _download_lazy.center()]).center()

    return (plot_ilines,)


@app.cell(hide_code=True)
def _():
    # plot_ilines(
    #     data=seis_np,
    #     idxs=(il_idx, il_idx_2),
    #     buttons     = (inline_full_num.value, inline_full_num_2.value),
    #     vertbuttons = (xline_full_num.value,  xline_full_num_2.value),
    #     horbuttons  = (depth_full_num.value,  depth_full_num_2.value),
    #     vertidxs    = (xl_idx, xl_idx_2),
    #     horidxs     = (z_idx, z_idx_2),
    #     dims        = (n_ilines, n_xlines, nsample),
    #     dim_start   = (58, 510, 0),
    #     title       = "KerryOriginal",)
    return


@app.cell(hide_code=True)
def _(mo, np, plt, save_fig_buf, seiscmap, vminmax):
    def plot_xlines(
        data        : np.ndarray,
        idxs        : tuple[int, int],
        buttons     : tuple[int, int],
        vertbuttons : tuple[int, int],
        horbuttons  : tuple[int, int],
        vertidxs    : tuple[int, int],
        horidxs     : tuple[int, int],
        dims        : tuple[int, int, int],
        dim_start   : tuple[int, int, int],
        title       : str = "KerryOriginal",
        minmax      : dict[str, float] = vminmax
    ) -> mo.Html:
        fig_xline, ax_xline = plt.subplots(1,2, figsize=(10,7), sharey=False, layout="compressed")
        _n = 75
        ax_xline[0].imshow(data[:,idxs[0],:].transpose(), seiscmap(), aspect="auto", **minmax)
        ax_xline[1].imshow(data[:,idxs[1],:].transpose(), seiscmap(), aspect="auto", **minmax)

        for _ax, _d in zip(ax_xline, buttons):
            _ax.set_ylabel("TWT (s)", fontsize=15)
            _ax.set_xlabel("INLINE", labelpad=10, fontsize=15)

        
            _ax.set_xticks(
                np.arange(0, dims[0], _n),
                np.arange(dim_start[0], dims[0]+dim_start[0], _n),
                rotation=90)
            _ax.set_yticks(
                np.arange(0, dims[2], _n),
                np.arange(0, dims[2]*4, _n*4)/1000,
                rotation=0)

            # _axT = _ax.secondary_xaxis('top')
            # _axT.set_xticks(
            #     np.arange(0, dims[0], _n),
            #     np.arange(dim_start[0], dims[0]+dim_start[0], _n),
            #     rotation=90)

            _axR = _ax.secondary_yaxis('right')
            _axR.set_yticks(
                np.arange(0, dims[2], _n),
                np.arange(0, dims[2]*4, _n*4)/1000,
                rotation=0)
        
            _ax.set_title(f"$CROSSLINE\ {_d}$", fontsize=20, style="italic", pad=10)
            _ax.axvline(vertidxs[0], color="red", label=f"INLINE {vertbuttons[0]}")
            _ax.axvline(vertidxs[1], color="red", linestyle="--", label=f"INLINE {vertbuttons[1]}")
            _ax.axhline(horidxs[0], color="black", label=f"TWT {horbuttons[0]} ms")
            _ax.axhline(horidxs[1], color="black", linestyle="--", label=f"TWT {horbuttons[1]} ms")
            _ax.grid(which="both", color="black", alpha=0.25, markevery=2, snap=True)
        
        _handles, _labels = ax_xline[1].get_legend_handles_labels()
        fig_xline.legend(_handles, _labels, ncols=4, loc='lower center', bbox_to_anchor =(0.5,-0.05))

        _fig_file_name = ["KerryOriginal", 
                          "XL", str(buttons[0])     + "-" + str(buttons[1]),
                          "IL", str(vertbuttons[0]) + "-" + str(vertbuttons[1]),
                          "TWT", str(horbuttons[0]) + "-" + str(horbuttons[1])
                         ]
        _fig_file_name = "_".join(_fig_file_name)
        _download_lazy = mo.download(
            data = save_fig_buf(f=fig_xline),
            filename = _fig_file_name,
            label = _fig_file_name
        )
        return mo.vstack([mo.as_html(plt.gcf()).center(), _download_lazy.center()]).center()

    return (plot_xlines,)


@app.cell(hide_code=True)
def _():
    # plot_xlines(
    #     data=seis_np,
    #     idxs=(xl_idx, xl_idx_2),
    #     buttons     = (xline_full_num.value,  xline_full_num_2.value),
    #     vertbuttons = (inline_full_num.value, inline_full_num_2.value),
    #     horbuttons  = (depth_full_num.value,  depth_full_num_2.value),
    #     vertidxs    = (il_idx, il_idx_2),
    #     horidxs     = (z_idx, z_idx_2),
    #     dims        = (n_ilines, n_xlines, nsample),
    #     dim_start   = (58, 510, 0),
    #     title       = "KerryOriginal",
    # )
    return


@app.cell(hide_code=True)
def _(mo, np, plt, save_fig_buf, seiscmap, vminmax):
    def plot_timedepths(
        data: np.ndarray,
        idxs: tuple[int, int],
        buttons     : tuple[int, int],
        vertbuttons : tuple[int, int],
        horbuttons  : tuple[int, int],
        vertidxs    : tuple[int, int],
        horidxs     : tuple[int, int],
        dims        : tuple[int, int, int],
        dim_start   : tuple[int, int, int],
        title       : str = "KerryOriginal",
        minmax      : dict[str, float|int] = vminmax,
    ) -> mo.Html:
        fig_depth, ax_depth = plt.subplots(1,2, figsize=(10,10), sharey=False, layout="compressed")
        ax_depth[0].imshow(data[:,:,idxs[0]].T, seiscmap(), aspect="equal", origin='upper', **minmax)
        ax_depth[1].imshow(data[:,:,idxs[1]].T, seiscmap(), aspect="equal", origin='upper', **minmax)
        _n = 25
        for _ax, _d in zip(ax_depth, buttons):
            _ax.set_xlabel("INLINE", fontsize=15)
            _ax.set_ylabel("CROSSLINE", fontsize=15)
        
            _ax.set_xticks(
                np.arange(0, dims[0], _n),
                np.arange(dim_start[0], dims[0]+dim_start[0], _n),
                rotation=90)
            _ax.set_yticks(
                np.arange(0, dims[1], _n),
                np.arange(dim_start[1], dims[1]+dim_start[1], _n),
                rotation=0)
        
            _axR = _ax.secondary_yaxis('right')
            _axT = _ax.secondary_xaxis('top')    

            _axT.set_xticks(
                np.arange(0, dims[0], _n),
                np.arange(dim_start[0], dims[0]+dim_start[0], _n),
                rotation=90)
            _axR.set_yticks(
                np.arange(0, dims[1], _n),
                np.arange(dim_start[1], dims[1]+dim_start[1], _n),
                rotation=0)

            _ax.axhline(vertidxs[0], color="blue", label=f"CROSSLINE {vertbuttons[0]}")
            _ax.axhline(vertidxs[1], color="blue", linestyle="--", label=f"CROSSLINE {vertbuttons[1]}")
            _ax.axvline(horidxs[0], color="red", label=f"INLINE {horbuttons[0]}")
            _ax.axvline(horidxs[1], color="red", linestyle="--", label=f"INLINE {horbuttons[1]}")
            _ax.set_title(f"$TIME\ SLICE\ {_d} ms$", style="italic", pad=10)
            _ax.grid(which="both", color="black", alpha=0.25, markevery=2, snap=True)

        _handles, _labels = ax_depth[1].get_legend_handles_labels()
        fig_depth.legend(_handles, _labels, ncols=4, loc='lower center', bbox_to_anchor =(0.5,-0.05))
        _fig_file_name = [title,
                          "TWT", str(buttons[0])     + "-" + str(buttons[1]),
                          "XL", str(vertbuttons[0]) + "-" + str(vertbuttons[1]),
                          "IL", str(horbuttons[0]) + "-" + str(horbuttons[1])
                         ]
        _fig_file_name = "_".join(_fig_file_name)
        # plt.show(

        _download_lazy = mo.download(
            data = save_fig_buf(plt.gcf()),
            filename = _fig_file_name,
            label = _fig_file_name
        )
        return mo.vstack([mo.as_html(plt.gcf()).center(), _download_lazy.center()]).center()
    return (plot_timedepths,)


@app.cell(hide_code=True)
def _():
    # plot_timedepths(
    #     data        = seis_np,
    #     idxs        = (z_idx, z_idx_2),
    #     buttons     = (depth_full_num.value,  depth_full_num_2.value),
    #     vertbuttons = (xline_full_num.value,  xline_full_num_2.value),
    #     horbuttons  = (inline_full_num.value, inline_full_num_2.value),
    #     vertidxs    = (xl_idx, xl_idx_2),
    #     horidxs     = (il_idx, il_idx_2),
    #     dims        = (n_ilines, n_xlines, nsample),
    #     dim_start   = (58, 510, 0),
    #     title       = "KerryOriginal",
    # )
    return


@app.cell(hide_code=True)
def _(MASK, filedir, mo, np, seis_np, time):
    def save_on_click(data=seis_np, suffix="", save_mask=False, mask=MASK):
        t0_4 = time.time()
        with mo.status.progress_bar(
            total=2 if save_mask else 1,
            title="Saving The Data",
            subtitle="Please wait",
        ) as _bar:
            _bar.update(subtitle="Saving Data")
            np.savez(filedir / f"kerry3d{suffix}", data)
            if save_mask:
                np.savez(filedir / f"kerry3d_mask{suffix}", mask.astype(np.int16))
                _bar.update(subtitle="Saving Mask")
        return mo.md(
            "## `segy` Saved to npz: data save in {:.1f} min".format(
                (time.time() - t0_4) / 60
            )
        ).center()
    return (save_on_click,)


@app.cell(hide_code=True)
def _(mo):
    run_button = mo.ui.run_button(label="Click me to save!")
    # run_button.center()
    return (run_button,)


@app.cell(hide_code=True)
def _(mo, run_button, save_on_click):
    if run_button.value:
        BUTTON_1 = save_on_click(save_mask=True)
    else:
        BUTTON_1 = mo.md("### Save Original Data!").center()
    return (BUTTON_1,)


@app.cell(column=1, hide_code=True)
def _():
    from matplotlib.ticker import ScalarFormatter
    return


@app.cell(hide_code=True)
def _(mo):
    hist_binsize = mo.ui.number(start=10, stop=500, step=5, value=100, label="Histogram BinSize")

    return (hist_binsize,)


@app.cell(hide_code=True)
def _(hist_binsize, mo):
    mo.vstack([mo.md(r"""# Data Preprocessing """).center(), hist_binsize.center()])
    return


@app.cell(hide_code=True)
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

    ori_latex = [
        mo.hstack([mo.md(r"""## Original""").center(),
        _download_lazy.center()]),
        mo.ui.text_area(_latex_tabel, rows=10, full_width=True, max_length=100),
    ]
    return (ori_latex,)


@app.cell(hide_code=True)
def _(mo, ori_latex, seis_stats):
    table_stat = mo.ui.table(data=seis_stats, pagination=False)
    mo.vstack(
        ori_latex + [
            mo.md("Select Statistic Values to be plot!").center(),
            table_stat
        ]
    )
    return (table_stat,)


@app.cell(hide_code=True)
def _():
    from matplotlib.ticker import FormatStrFormatter
    return (FormatStrFormatter,)


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
    return (make_histogram_seis,)


@app.cell
def _(make_histogram_seis, plt):
    fig_stat, ax_stat = plt.subplots(1,1, figsize=(5,5))
    N, bins, patches = make_histogram_seis()
    return ax_stat, fig_stat, patches


@app.cell(hide_code=True)
def _(
    FormatStrFormatter,
    SEIS_MAX,
    SEIS_MIN,
    ax_stat,
    cycle,
    fig_stat,
    hist_binsize,
    mo,
    np,
    patches,
    save_fig_buf,
    table_stat,
):
    line_colors_stats = cycle(["r", 'k', 'b', 'g', 'orange'])

    n_stats_to_plot = len(table_stat.value)
    ax_stat.clear(); ax_stat.add_patch(patches)
    _xlabel = np.linspace(SEIS_MIN-1, SEIS_MAX+1, 9)
    _fig_file_name = ["KerryOriginal", str(hist_binsize.value)]

    if n_stats_to_plot != 0:
        for _s, _c in zip(table_stat.value, line_colors_stats):
            ax_stat.axvline(_s["Value"],
                            color=_c, linestyle="dashed",
                            linewidth=1,
                            label=f"${_s['Statistic'].title()}={_s['Value']:.3f}$")
            _fig_file_name += [_s['Statistic']]
        ax_stat.legend(ncols=1 + n_stats_to_plot//2, bbox_to_anchor =(0.5,-0.37), loc='lower center')
        _title = "Histogram of Seismic Amplitude"
        ax_stat.set_ylabel("Frequency", fontstyle="italic")
        ax_stat.set_xlabel("Amplitude", fontstyle="italic")
        # ax_stat.set_xticks()
    else:
        _title = "Histogram of Seismic' Amplitude and Pick Statistic Values to plot, btw!"

    ax_stat.grid(which="both",alpha=0.25)
    ax_stat.autoscale_view()
    ax_stat.set_xticks(_xlabel, _xlabel, rotation=90)
    ax_stat.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
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
    mo.md(r"""## Cropping & Standardization""").center()
    return


@app.cell
def _(SEIS_MEAN, SEIS_STD, masked_crop_data, np):
    seis_flatten_stdr_ori = (masked_crop_data - SEIS_MEAN)/(SEIS_STD)
    seis_flatten_stdr = seis_flatten_stdr_ori[~np.isnan(seis_flatten_stdr_ori)].flatten()
    SEIS_MAX_stdr = np.nanmax(seis_flatten_stdr)
    SEIS_MIN_stdr = np.nanmin(seis_flatten_stdr)
    seis_stats_stdr = {}
    seis_stats_stdr['max'] = SEIS_MAX_stdr
    seis_stats_stdr['min'] = SEIS_MIN_stdr
    seis_stats_stdr['std'] = np.nanstd(seis_flatten_stdr)
    seis_stats_stdr['mean'] = np.nanmean(seis_flatten_stdr)
    seis_stats_stdr['median'] = np.nanmedian(seis_flatten_stdr)
    seis_stats_stdr = [dict(Statistic=stat, Value=val) for stat, val in seis_stats_stdr.items()]
    return seis_flatten_stdr, seis_stats_stdr


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

    std_latex = [
        mo.hstack([mo.md(r"""## Cropped & Standardization""").center(),
        _download_lazy.center()]),
        mo.ui.text_area(_latex_tabel, rows=10, full_width=True, max_length=100),
    ]
    return (std_latex,)


@app.cell
def _(mo, seis_stats_stdr, std_latex):
    table_stat_stdr = mo.ui.table(data=seis_stats_stdr, pagination=False)

    mo.vstack(
        std_latex + [
            mo.md("Select Statistic Values to be plot!").center(),
            table_stat_stdr
        ]
    )
    return (table_stat_stdr,)


@app.cell
def _():
    return


@app.cell
def _(make_histogram_seis, plt, seis_flatten_stdr):
    fig_stat_stdr, ax_stat_stdr = plt.subplots(1,1, figsize=(5,5))
    N_stdr, bins_stdr, patches_stdr = make_histogram_seis(data=seis_flatten_stdr)
    return ax_stat_stdr, fig_stat_stdr, patches_stdr


@app.cell(hide_code=True)
def _(
    FormatStrFormatter,
    SEIS_MAX,
    SEIS_MIN,
    ax_stat_stdr,
    fig_stat_stdr,
    hist_binsize,
    line_colors_stats,
    mo,
    np,
    patches_stdr,
    plt,
    save_fig_buf,
    table_stat_stdr,
):
    n_stats_to_plot_stdr = len(table_stat_stdr.value)
    ax_stat_stdr.clear();
    ax_stat_stdr.add_patch(patches_stdr)
    _fig_file_name = ["Kerrystandardized_cropped", str(hist_binsize.value)]

    _xlabel = np.linspace(SEIS_MIN-1, SEIS_MAX+1, 9)
    if n_stats_to_plot_stdr != 0:
        for _s, _c in zip(table_stat_stdr.value, line_colors_stats):
            ax_stat_stdr.axvline(_s["Value"],
                            color=_c, linestyle="dashed",
                            linewidth=1,
                            label=f"${_s['Statistic'].title()}={_s['Value']:.3f}$")
            _fig_file_name += [_s['Statistic']]
        ax_stat_stdr.legend(ncols=1 + n_stats_to_plot_stdr//2, bbox_to_anchor =(0.5,-0.35), loc='lower center')
        _title = "Histogram of Seismic Amplitude"
        ax_stat_stdr.set_ylabel("Frequency", fontstyle="italic")
        ax_stat_stdr.set_xlabel("Amplitude", fontstyle="italic")
    else:
        _title = "Histogram of Seismic' Amplitude and Pick Statistic Values to plot, btw!"

    ax_stat_stdr.grid(which="both",alpha=0.25)
    ax_stat_stdr.autoscale_view()
    ax_stat_stdr.set_xticks(_xlabel, _xlabel, rotation=90)
    ax_stat_stdr.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    plt.tight_layout()
    _fig_file_name = "_".join(_fig_file_name)
    _download_lazy = mo.download(
        data = save_fig_buf(f=fig_stat_stdr),
        filename = _fig_file_name,
        label = _fig_file_name
    )
    # print(_xlabel)
    mo.vstack([mo.as_html(fig_stat_stdr).center(), _download_lazy.center()]).center()
    # plt.show()        

    return


@app.cell(column=2, hide_code=True)
def _(mo):
    mo.md(r"""# Cropping Data """).center()
    return


@app.cell(hide_code=True)
def _(IL_END, IL_START, inline_full_num, inline_full_num_2):
    if inline_full_num.value < inline_full_num_2.value:
        il_crop_start = inline_full_num.value
        il_crop_end   = inline_full_num_2.value
    elif inline_full_num.value > inline_full_num_2.value:
        il_crop_start = inline_full_num_2.value
        il_crop_end   = inline_full_num.value
    else:
        il_crop_start = IL_START
        il_crop_end   = IL_END

    idx_il_crop_start  = il_crop_start - IL_START
    idx_il_crop_stop   = il_crop_end   - idx_il_crop_start
    # print((il_crop_start, il_crop_end), (idx_il_crop_start, idx_il_crop_stop))
    return idx_il_crop_start, il_crop_end, il_crop_start


@app.cell(hide_code=True)
def _(XL_END, XL_START, xline_full_num, xline_full_num_2):
    if xline_full_num.value < xline_full_num_2.value:
        xl_crop_start = xline_full_num.value
        xl_crop_end   = xline_full_num_2.value
    elif xline_full_num.value > xline_full_num_2.value:
        xl_crop_start = xline_full_num_2.value
        xl_crop_end   = xline_full_num.value

    else:
        xl_crop_start = XL_START
        xl_crop_end   = XL_END

    idx_xl_crop_start  = xl_crop_start - XL_START
    idx_xl_crop_stop   = xl_crop_end   - idx_xl_crop_start
    # print((xl_crop_start, xl_crop_end), (idx_xl_crop_start, idx_xl_crop_stop))
    return idx_xl_crop_start, xl_crop_end, xl_crop_start


@app.cell(hide_code=True)
def _(Z_END, Z_START, depth_full_num, depth_full_num_2):
    if depth_full_num.value < depth_full_num_2.value:
        depth_crop_start = depth_full_num.value
        depth_crop_end   = depth_full_num_2.value
    elif depth_full_num.value > depth_full_num_2.value:
        depth_crop_start = depth_full_num_2.value
        depth_crop_end   = depth_full_num.value

    else:
        depth_crop_start = Z_START
        depth_crop_end   = Z_END


    idx_depth_crop_start  = depth_crop_start - Z_START
    idx_depth_crop_stop   = depth_crop_end   - idx_depth_crop_start
    # print((depth_crop_start, depth_crop_end), (idx_depth_crop_start, idx_depth_crop_stop))
    return depth_crop_end, depth_crop_start, idx_depth_crop_start


@app.cell
def _(np, seis_np):
    template_zero = np.zeros_like(seis_np)
    return (template_zero,)


@app.cell(hide_code=True)
def _(
    IL_START,
    XL_START,
    depth_crop_end,
    depth_crop_start,
    il_crop_end,
    il_crop_start,
    np,
    sample_rate,
    seis_np,
    template_zero,
    xl_crop_end,
    xl_crop_start,
):
    # normalized_seis_np = ((seis_np - SEIS_MIN) / (SEIS_MAX - SEIS_MIN))
    # normalized_seis_np = 2 * ((normalized_seis_np - 0) / (1 - 0)) - 1

    # normalized_seis_np = 2 * ((seis_np - SEIS_MIN) / (SEIS_MAX - SEIS_MIN)) - 1

    # normalized_seis_np = (seis_np - SEIS_MEAN) / SEIS_STD

    normalized_seis_np = seis_np

    masked_crop_data = template_zero * np.nan
    # masked_crop_data *= np.nan
    masked_crop_data[
            slice(il_crop_start - IL_START, il_crop_end - IL_START),
            slice(xl_crop_start - XL_START, xl_crop_end - XL_START),
            slice(depth_crop_start//sample_rate, depth_crop_end//sample_rate)] = normalized_seis_np[
                slice(il_crop_start - IL_START, il_crop_end - IL_START),
                slice(xl_crop_start - XL_START, xl_crop_end - XL_START),
                slice(depth_crop_start//sample_rate, depth_crop_end//sample_rate)
                ]
    return (masked_crop_data,)


@app.cell(hide_code=True)
def _(
    depth_full_num,
    depth_full_num_2,
    il_idx,
    il_idx_2,
    inline_full_num,
    inline_full_num_2,
    masked_crop_data,
    mo,
    n_ilines,
    n_xlines,
    nsample,
    plot_ilines,
    seis_np,
    xl_idx,
    xl_idx_2,
    xline_full_num,
    xline_full_num_2,
    z_idx,
    z_idx_2,
):
    mo.hstack([
        mo.vstack([
            mo.md("## Original").center(),
            plot_ilines(data=seis_np,
                        idxs=(il_idx, il_idx_2),
                        buttons     = (inline_full_num.value, inline_full_num_2.value),
                        vertbuttons = (xline_full_num.value,  xline_full_num_2.value),
                        horbuttons  = (depth_full_num.value,  depth_full_num_2.value),
                        dims        = (n_ilines, n_xlines, nsample),
                        vertidxs    = (xl_idx, xl_idx_2),
                        horidxs     = (z_idx, z_idx_2),
                        dim_start   = (58, 510, 0),
                        title = "KerryOriginal",)]),
        mo.vstack([
            mo.md("## Cropped").center(),
            plot_ilines(data = masked_crop_data,
                        buttons     = (inline_full_num.value, inline_full_num_2.value),
                        vertbuttons = (xline_full_num.value,  xline_full_num_2.value),
                        horbuttons  = (depth_full_num.value,  depth_full_num_2.value),
                        dims        = (n_ilines, n_xlines, nsample),
                        vertidxs    = (xl_idx, xl_idx_2),
                        horidxs     = (z_idx, z_idx_2),
                        dim_start   = (58, 510, 0),
                        idxs = (il_idx, il_idx_2-1),
                        title = "KerryCropped",
                        # minmax = dict(vmin=-1, vmax=1)
                       )])
    ])
    return


@app.cell(hide_code=True)
def _(
    depth_full_num,
    depth_full_num_2,
    il_idx,
    il_idx_2,
    inline_full_num,
    inline_full_num_2,
    masked_crop_data,
    mo,
    n_ilines,
    n_xlines,
    nsample,
    plot_xlines,
    seis_np,
    xl_idx,
    xl_idx_2,
    xline_full_num,
    xline_full_num_2,
    z_idx,
    z_idx_2,
):
    mo.hstack([
        mo.vstack([
            mo.md("## Original").center(),
            plot_xlines(data=seis_np,
                        idxs=(xl_idx, xl_idx_2),
                        buttons     = (depth_full_num.value,  depth_full_num_2.value),
                        vertbuttons = (xline_full_num.value,  xline_full_num_2.value),
                        horbuttons  = (inline_full_num.value, inline_full_num_2.value),
                        dims        = (n_ilines, n_xlines, nsample),
                        vertidxs    = (il_idx, il_idx_2),
                        horidxs     = (z_idx, z_idx_2),
                        dim_start   = (58, 510, 0),                    
                        title = "KerryOriginal",)]),
        mo.vstack([
            mo.md("## Cropped").center(),
            plot_xlines(data = masked_crop_data,
                        idxs = (xl_idx, xl_idx_2-1),
                        buttons     = (depth_full_num.value,  depth_full_num_2.value),
                        vertbuttons = (xline_full_num.value,  xline_full_num_2.value),
                        horbuttons  = (inline_full_num.value, inline_full_num_2.value),
                        dims        = (n_ilines, n_xlines, nsample),
                        vertidxs    = (il_idx, il_idx_2),
                        horidxs     = (z_idx, z_idx_2),
                        dim_start   = (58, 510, 0),                    
                        title       = "KerryCropped",
                        # minmax       = dict(vmin=-1, vmax=1) 
                       )])
    ])
    return


@app.cell(hide_code=True)
def _():
    # seis_kerry = np.load("./kerry_from_github.npy")
    # seis_kerry_r = 2 * ((seis_kerry - np.nanmin(seis_kerry)) / (np.nanmax(seis_kerry) - np.nanmin(seis_kerry))) - 1
    # _sum_ver = np.sum(np.abs(seis_kerry_r), axis=2)
    # _sum_ver = np.where(_sum_ver >= _sum_ver.min()+ 0.01, True, False)
    # for _i in range(272):
    #     _slice = seis_kerry_r[:, :, _i]
    #     seis_kerry[:, :, _i] = np.where(_sum_ver, _slice, np.nan)
    return


@app.cell(hide_code=True)
def _():
    # _x, _y, _z = seis_kerry.shape
    # mo.md(f"""
    # | LINE      | START                   | STOP                  | RANGE                                      | CROPPED GITHUB | 
    # | :-------- | :---------------------: | :-------------------: | :----------------------------------------: | :------------: | 
    # | INLINE    | ${il_crop_start}$       | ${il_crop_end}$       | ${il_crop_end - il_crop_start}$            | ${_y}$         | 
    # | CROSSLINE | ${xl_crop_start}$       | ${xl_crop_end}$       | ${xl_crop_end - xl_crop_start}$            | ${_x}$         | 
    # | TWT       | ${depth_crop_start//4}$ | ${depth_crop_end//4}$ | ${(depth_crop_end - depth_crop_start)//4}$ | ${_z}$         | 
    # """).center()
    return


@app.cell(hide_code=True)
def _():
    # seis_kerry_shape = seis_kerry.shape
    # kerry_ghub_z = mo.ui.number(value=10, start=0, stop=seis_kerry_shape[2]-1, step=1, label="z")
    # kerry_ghub_x = mo.ui.number(value=34, start=0, stop=seis_kerry_shape[0]-1, step=1, label="x")
    # kerry_ghub_y = mo.ui.number(value=188, start=0, stop=seis_kerry_shape[1]-1, step=1, label="y")
    return


@app.cell(hide_code=True)
def _():
    # _f, _ax = plt.subplot_mosaic(
    #     [
    #         # ["z", "z", "z", "x", "x"],
    #         # ["y", 'y', 'y', ".", "."],
    #         ["x", "x", "y", "y", "y"],
    #         ["x", "x", "y", "y", "y"],
    #         ["z", "z", "z", "z", "z"],
    #     ],
    #     figsize=(15, 10),
    #     gridspec_kw={"width_ratios":[0.5, 1, 0.5, 1.2, 3]},#, "height_ratios":[0.5, 0.5, 2]},
    #     layout="compressed")
    # _dict = dict(cmap=seiscmap(), aspect="auto", vmin=np.nanmin(seis_kerry), vmax=np.nanmax(seis_kerry))
    # _z = _ax['z'].imshow(seis_kerry[:,:,kerry_ghub_z.value].T  , **_dict, origin="upper")
    # _ax['x'].imshow(seis_kerry[kerry_ghub_x.value,:,:].T, **_dict)
    # _ax['y'].imshow(seis_kerry[:,kerry_ghub_y.value,:].T  , **_dict, origin="upper")
    # plt.colorbar(_z, orientation="horizontal")
    # _ax['y'].grid(color="yellow")
    # _ax['x'].grid(color="yellow")
    # mo.vstack(
    #     [
    #         mo.hstack(
    #             [kerry_ghub_x, kerry_ghub_y, kerry_ghub_z]
    #         ),
    #         mo.as_html(plt.gcf()).center()
    #     ])
    return


@app.cell(hide_code=True)
def _():
    # cropped_norm_ori = normalized_seis_np[
    #         slice(il_crop_start - IL_START, il_crop_end - IL_START),
    #         slice(xl_crop_start - XL_START, xl_crop_end - XL_START),
    #         slice(depth_crop_start//sample_rate, 1+(depth_crop_end//sample_rate)),
    #     ]
    return


@app.cell(hide_code=True)
def _():
    # _n = 0.1
    # _f, _ax = plt.subplot_mosaic(
    #     [
    #         # ["z", "z", "z", "x", "x"],
    #         # ["y", 'y', 'y', ".", "."],
    #         ["x", "x", "y", "y", "y"],
    #         ["x", "x", "y", "y", "y"],
    #         ["z", "z", "z", "z", "z"],
    #     ],
    #     figsize=(15, 10),
    #     gridspec_kw={"width_ratios":[0.5, 1, 0.5, 1.2, 3]},#, "height_ratios":[0.5, 0.5, 2]},
    #     layout="compressed")
    # _dict = dict(cmap=seiscmap(), aspect="auto", vmin=-1, vmax=1)
    # _z = _ax['z'].imshow(cropped_norm_ori[:,:,kerry_ghub_z.value]  , **_dict, origin="upper")
    # _ax['y'].imshow(cropped_norm_ori[kerry_ghub_y.value,:,:].T, **_dict)
    # _ax['x'].imshow(cropped_norm_ori[:,kerry_ghub_x.value,:].T  , **_dict, origin="upper")
    # _ax['y'].grid(color="yellow")
    # _ax['x'].grid(color="yellow")
    # plt.colorbar(_z, orientation="horizontal")
    # mo.vstack(
    #     [
    #         mo.hstack(
    #             [kerry_ghub_x, kerry_ghub_y, kerry_ghub_z]
    #         ),
    #         mo.as_html(plt.gcf()).center()
    #     ])
    return


@app.cell(hide_code=True)
def _(
    depth_full_num,
    depth_full_num_2,
    il_idx,
    il_idx_2,
    inline_full_num,
    inline_full_num_2,
    masked_crop_data,
    mo,
    n_ilines,
    n_xlines,
    nsample,
    plot_timedepths,
    seis_np,
    xl_idx,
    xl_idx_2,
    xline_full_num,
    xline_full_num_2,
    z_idx,
    z_idx_2,
):
    mo.hstack([
        mo.vstack([
            mo.md("## Original").center(),
            plot_timedepths(
                data        = seis_np,
                idxs        = (z_idx, z_idx_2),
                buttons     = (depth_full_num.value,  depth_full_num_2.value),
                vertbuttons = (xline_full_num.value,  xline_full_num_2.value),
                horbuttons  = (inline_full_num.value, inline_full_num_2.value),
                dims        = (n_ilines, n_xlines, nsample),
                vertidxs    = (xl_idx, xl_idx_2),
                horidxs     = (il_idx, il_idx_2),
                dim_start   = (58, 510, 0),
                title       = "KerryOriginal",
            )]),
        mo.vstack([
            mo.md("## Cropped").center(),
            plot_timedepths(
                data = masked_crop_data,
                idxs = (z_idx, z_idx_2-1),
                buttons     = (depth_full_num.value,  depth_full_num_2.value),
                vertbuttons = (xline_full_num.value,  xline_full_num_2.value),
                horbuttons  = (inline_full_num.value, inline_full_num_2.value),
                vertidxs    = (xl_idx, xl_idx_2),
                horidxs     = (il_idx, il_idx_2),
                dims        = (n_ilines, n_xlines, nsample),
                dim_start   = (58, 510, 0),
                title = "KerryCropped", 
                # minmax = {"vmin":-1, "vmax":1},
            )])
    ])
    return


@app.cell(hide_code=True)
def _(
    IL_START,
    XL_START,
    depth_crop_end,
    depth_crop_start,
    il_crop_end,
    il_crop_start,
    masked_crop_data,
    np,
    sample_rate,
    xl_crop_end,
    xl_crop_start,
):
    to_be_saved_cropped = masked_crop_data[slice(il_crop_start - IL_START, il_crop_end - IL_START), :, :]
    to_be_saved_cropped = to_be_saved_cropped[:, slice(xl_crop_start - XL_START, xl_crop_end - XL_START), :]
    to_be_saved_cropped = to_be_saved_cropped[:, :, slice(depth_crop_start//sample_rate, depth_crop_end//sample_rate)]

    mask_tb_cropped = to_be_saved_cropped.astype(np.bool_)

    # print(to_be_saved_cropped.shape, mask_tb_cropped.shape)
    return mask_tb_cropped, to_be_saved_cropped


@app.cell(hide_code=True)
def _(np, to_be_saved_cropped):
    seis_flatten_crop = to_be_saved_cropped.flatten().flatten().flatten()
    SEIS_CROP_MEAN = np.nanmean(seis_flatten_crop)
    SEIS_CROP_STD = np.nanstd(seis_flatten_crop)
    seis_flatten_crop = (seis_flatten_crop[~np.isnan(seis_flatten_crop)])# - SEIS_CROP_MEAN) / SEIS_CROP_STD
    SEIS_MAX_stdr_crop = np.nanmax(seis_flatten_crop)
    SEIS_MIN_stdr_crop = np.nanmin(seis_flatten_crop)
    seis_stats_stdr_crop = {}
    seis_stats_stdr_crop['max'] = SEIS_MAX_stdr_crop
    seis_stats_stdr_crop['min'] = SEIS_MIN_stdr_crop
    seis_stats_stdr_crop['std'] = SEIS_CROP_STD
    seis_stats_stdr_crop['mean'] = SEIS_CROP_MEAN
    seis_stats_stdr_crop['median'] = np.nanmedian(seis_flatten_crop)
    seis_stats_stdr_crop = [dict(Statistic=stat, Value=val) for stat, val in seis_stats_stdr_crop.items()]
    return seis_flatten_crop, seis_stats_stdr_crop


@app.cell(hide_code=True)
def _(mo, pd, save_tex_buf, seis_stats_stdr_crop):
    _df = pd.DataFrame(seis_stats_stdr_crop)
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
    croped_latex = [
        mo.hstack([mo.md(r"""## Cropped""").center(),
        _download_lazy.center()]),
        mo.ui.text_area(_latex_tabel, rows=10, full_width=True, max_length=100),
    ]
    return (croped_latex,)


@app.cell(hide_code=True)
def _(croped_latex, mo, seis_stats_stdr_crop):
    table_stat_crop = mo.ui.table(data=seis_stats_stdr_crop, pagination=False)

    mo.vstack(
        croped_latex + [
            mo.md("Select Statistic Values to be plot!").center(),
            table_stat_crop
        ]
    )
    return (table_stat_crop,)


@app.cell(hide_code=True)
def _(make_histogram_seis, plt, seis_flatten_crop):
    fig_stat_crop, ax_stat_crop = plt.subplots(1,1, figsize=(5,5))
    N_crop, bins_crop, patches_crop = make_histogram_seis(data=seis_flatten_crop)
    return ax_stat_crop, fig_stat_crop, patches_crop


@app.cell(hide_code=True)
def _(
    FormatStrFormatter,
    SEIS_MAX,
    SEIS_MIN,
    ax_stat_crop,
    fig_stat_crop,
    hist_binsize,
    line_colors_stats,
    mo,
    np,
    patches_crop,
    plt,
    save_fig_buf,
    table_stat_crop,
):
    n_stats_to_plot_crop = len(table_stat_crop.value)
    ax_stat_crop.clear();
    ax_stat_crop.add_patch(patches_crop)
    _fig_file_name = ["Kerry_cropped", str(hist_binsize.value)]

    _xlabel = np.linspace(SEIS_MIN-1, SEIS_MAX+1, 9)
    if n_stats_to_plot_crop != 0:
        for _s, _c in zip(table_stat_crop.value, line_colors_stats):
            ax_stat_crop.axvline(_s["Value"],
                            color=_c, linestyle="dashed",
                            linewidth=1,
                            label=f"${_s['Statistic'].title()}={_s['Value']:.3f}$")
            _fig_file_name += [_s['Statistic']]
        ax_stat_crop.legend(ncols=1 + n_stats_to_plot_crop//2, bbox_to_anchor =(0.5,-0.35), loc='lower center')
        _title = "Histogram of Seismic Amplitude"
        ax_stat_crop.set_ylabel("Frequency", fontstyle="italic")
        ax_stat_crop.set_xlabel("Amplitude", fontstyle="italic")
    else:
        _title = "Histogram of Seismic' Amplitude and Pick Statistic Values to plot, btw!"

    ax_stat_crop.grid(which="both",alpha=0.25)
    ax_stat_crop.autoscale_view()
    ax_stat_crop.set_xticks(_xlabel, _xlabel, rotation=90)
    ax_stat_crop.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    plt.tight_layout()
    _fig_file_name = "_".join(_fig_file_name)
    _download_lazy = mo.download(
        data = save_fig_buf(f=fig_stat_crop),
        filename = _fig_file_name,
        label = _fig_file_name
    )
    mo.vstack([mo.as_html(fig_stat_crop).center(), _download_lazy.center()]).center()

    return


@app.cell(hide_code=True)
def _(mo):
    run_button_cropped = mo.ui.run_button(label="Click me to save!")
    # run_button.center()
    return (run_button_cropped,)


@app.cell(hide_code=True)
def _(
    depth_crop_end,
    depth_crop_start,
    il_crop_end,
    il_crop_start,
    mask_tb_cropped,
    mo,
    np,
    run_button_cropped,
    save_on_click,
    to_be_saved_cropped,
    xl_crop_end,
    xl_crop_start,
):
    _buttons     = (depth_crop_start, depth_crop_end)
    _vertbuttons = (xl_crop_start, xl_crop_end)
    _horbuttons  = (il_crop_start, il_crop_end)

    _suffix      = "_".join(
        [
            "_TWT", str(_buttons[0])     + "-" + str(_buttons[1]),
            "XL", str(_vertbuttons[0]) + "-" + str(_vertbuttons[1]),
            "IL", str(_horbuttons[0]) + "-" + str(_horbuttons[1])
        ])
    # print(_suffix)
    if run_button_cropped.value:
        BUTTON_CROPPED = save_on_click(data=to_be_saved_cropped, suffix=_suffix, save_mask=True, mask=mask_tb_cropped.astype(np.int16))
    else:
        BUTTON_CROPPED = mo.md("### Save Cropped data to `.npz`").center()
    return (BUTTON_CROPPED,)


@app.cell(hide_code=True)
def _(BUTTON_1, BUTTON_CROPPED, mo, run_button, run_button_cropped):
    mo.hstack([
        mo.vstack([
            BUTTON_1,
            run_button.center(),
        ]),
        mo.vstack([
            BUTTON_CROPPED,
            run_button_cropped.center(),
        ])
    ])
    return


@app.cell
def _():
    return


@app.cell(column=3, hide_code=True)
def _(
    depth_crop_end,
    depth_crop_start,
    il_crop_end,
    il_crop_start,
    mo,
    sample_rate,
    xl_crop_end,
    xl_crop_start,
):
    _buttons     = (depth_crop_start, depth_crop_start)
    _vertbuttons = (xl_crop_start, xl_crop_end)
    _horbuttons  = (il_crop_start, il_crop_end)

    boundary_cropped = mo.md("""
    | Boundary  | Start | Stop  |
    | :-------: | :---: | :--:  |
    | CROSSLINE | {xl1} | {xl2} |
    | INLINE    | {il1} | {il2} |
    | DEPTH     | {z1}  | {z2}  |
    """).center().batch(
        xl1 = mo.ui.number(start=xl_crop_start, stop=xl_crop_end, step=1, label="", value=xl_crop_start),
        xl2 = mo.ui.number(start=xl_crop_start, stop=xl_crop_end, step=1, label="", value=xl_crop_end-1),
        il1 = mo.ui.number(start=il_crop_start, stop=il_crop_end, step=1, label="", value=il_crop_start,),
        il2 = mo.ui.number(start=il_crop_start, stop=il_crop_end, step=1, label="", value=il_crop_end - 1,),
        z1 = mo.ui.number(start=depth_crop_start,  stop=depth_crop_end,  step=sample_rate, label="", value=depth_crop_start),
        z2 = mo.ui.number(start=depth_crop_start,  stop=depth_crop_end,  step=sample_rate, label="", value=depth_crop_end - 1),
    ).form()
    boundary_cropped
    return (boundary_cropped,)


@app.cell
def _(xl_crop_end):
    xl_crop_end
    return


@app.cell(hide_code=True)
def _(TpVal, boundary_cropped):
    xline_full_crp  = TpVal(boundary_cropped.value["xl1"]) 
    inline_full_crp = TpVal(boundary_cropped.value["il1"]) 
    depth_full_crp  = TpVal(boundary_cropped.value["z1"]) 

    xline_full_crp_2  = TpVal(boundary_cropped.value["xl2"])
    inline_full_crp_2 = TpVal(boundary_cropped.value["il2"])
    depth_full_crp_2  = TpVal(boundary_cropped.value["z2"])
    return (
        depth_full_crp,
        depth_full_crp_2,
        inline_full_crp,
        inline_full_crp_2,
        xline_full_crp,
        xline_full_crp_2,
    )


@app.cell
def _(idx_depth_crop_start):
    idx_depth_crop_start
    return


@app.cell(hide_code=True)
def _(
    IL_START,
    XL_START,
    Z_START,
    depth_full_crp,
    depth_full_crp_2,
    idx_depth_crop_start,
    idx_il_crop_start,
    idx_xl_crop_start,
    inline_full_crp,
    inline_full_crp_2,
    xline_full_crp,
    xline_full_crp_2,
):
    il_idx_crp = inline_full_crp.value - IL_START - idx_il_crop_start
    xl_idx_crp =  xline_full_crp.value - XL_START - idx_xl_crop_start
    z_idx_crp  = ((depth_full_crp.value - Z_START) // 4) -  (idx_depth_crop_start // 4)

    il_idx_crp_2 = inline_full_crp_2.value - IL_START - idx_il_crop_start
    xl_idx_crp_2 =  xline_full_crp_2.value - XL_START - idx_xl_crop_start
    z_idx_crp_2  = ((depth_full_crp_2.value - Z_START) // 4) - (idx_depth_crop_start // 4)

    print(f"""

    {il_idx_crp=}
    {il_idx_crp_2=}

    {xl_idx_crp=}
    {xl_idx_crp_2=}

    {z_idx_crp=}
    {z_idx_crp_2=}

    """)
    return (
        il_idx_crp,
        il_idx_crp_2,
        xl_idx_crp,
        xl_idx_crp_2,
        z_idx_crp,
        z_idx_crp_2,
    )


@app.cell
def _(to_be_saved_cropped):
    cropped_shape = to_be_saved_cropped.shape
    return (cropped_shape,)


@app.cell(hide_code=True)
def _(
    cropped_shape,
    depth_full_crp,
    depth_full_crp_2,
    il_idx_crp,
    il_idx_crp_2,
    inline_full_crp,
    inline_full_crp_2,
    mo,
    plot_ilines,
    to_be_saved_cropped,
    xl_idx_crp,
    xl_idx_crp_2,
    xline_full_crp,
    xline_full_crp_2,
    z_idx_crp,
    z_idx_crp_2,
):
    mo.vstack([
        mo.md("## Cropped").center(),
        plot_ilines(data=to_be_saved_cropped,
                    idxs=(il_idx_crp, il_idx_crp_2),
                    buttons     = (inline_full_crp.value, inline_full_crp_2.value),
                    vertbuttons = (xline_full_crp.value,  xline_full_crp_2.value),
                    vertidxs    = (xl_idx_crp, xl_idx_crp_2),  
                    horbuttons  = (depth_full_crp.value,  depth_full_crp_2.value),
                    horidxs     = (z_idx_crp, z_idx_crp_2),
                    dims        = cropped_shape,
                    dim_start   = (
                        inline_full_crp.value + 0,
                        xline_full_crp.value + 0,
                        (depth_full_crp.value) + 0,
                    ),
                    title = "KerryCropped",)
    ])
    return


@app.cell(hide_code=True)
def _(
    cropped_shape,
    depth_full_crp,
    depth_full_crp_2,
    il_idx_crp,
    il_idx_crp_2,
    inline_full_crp,
    inline_full_crp_2,
    mo,
    plot_xlines,
    to_be_saved_cropped,
    xl_idx_crp,
    xl_idx_crp_2,
    xline_full_crp,
    xline_full_crp_2,
    z_idx_crp,
    z_idx_crp_2,
):
    mo.vstack([
        mo.md("## Cropped").center(),
        plot_xlines(data=to_be_saved_cropped,
                    idxs=(xl_idx_crp, xl_idx_crp_2),
                    buttons     = (xline_full_crp.value,  xline_full_crp_2.value),
                    vertbuttons = (inline_full_crp.value, inline_full_crp_2.value),
                    horbuttons  = (depth_full_crp.value,  depth_full_crp_2.value),
                    vertidxs    = (il_idx_crp, il_idx_crp_2),  
                    horidxs     = (z_idx_crp, z_idx_crp_2),
                    dims        = cropped_shape,
                    dim_start   = (
                        inline_full_crp.value + 0,
                        xline_full_crp.value + 0,
                        (depth_full_crp.value) + 0,
                    ),
                    title = "KerryCropped",)
    ])
    return


@app.cell(hide_code=True)
def _(
    cropped_shape,
    depth_full_crp,
    depth_full_crp_2,
    il_idx_crp,
    il_idx_crp_2,
    inline_full_crp,
    inline_full_crp_2,
    mo,
    plot_timedepths,
    to_be_saved_cropped,
    xl_idx_crp,
    xl_idx_crp_2,
    xline_full_crp,
    xline_full_crp_2,
    z_idx_crp,
    z_idx_crp_2,
):
    mo.vstack([
        mo.md("## Cropped").center(),
        plot_timedepths(
            data=to_be_saved_cropped,
            idxs=(z_idx_crp, z_idx_crp_2),
            buttons     = (depth_full_crp.value,  depth_full_crp_2.value),
            horbuttons  = (inline_full_crp.value, inline_full_crp_2.value),
            vertbuttons = (xline_full_crp.value,  xline_full_crp_2.value),
            horidxs     = (il_idx_crp, il_idx_crp_2),
            vertidxs    = (xl_idx_crp, xl_idx_crp_2),
            dims        = cropped_shape,
            dim_start   = (
                        inline_full_crp.value + 0,
                        xline_full_crp.value + 0,
                        (depth_full_crp.value) + 0,
            ),
            title = "KerryCropped",)
    ])
    return


if __name__ == "__main__":
    app.run()
