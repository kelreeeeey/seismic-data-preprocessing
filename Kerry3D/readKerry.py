import marimo

__generated_with = "0.13.11"
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


@app.cell
def _():
    import xarray as xr
    from seiscm import seismic as seiscmap
    from io import BytesIO, StringIO
    import pandas as pd
    from matplotlib import colors
    import matplotlib.patches as pltPatches
    import matplotlib.path as pltPath
    from matplotlib.ticker import FormatStrFormatter, StrMethodFormatter, Formatter

    from itertools import cycle
    from collections import namedtuple

    from pyproj import CRS as pjCRS
    import cartopy.crs as ccrs
    import shapely as shp
    from matplotlib_map_utils import north_arrow
    from matplotlib_scalebar.scalebar import ScaleBar
    from typing import Any
    return (
        Any,
        BytesIO,
        FormatStrFormatter,
        ScaleBar,
        ccrs,
        cycle,
        namedtuple,
        north_arrow,
        pd,
        pjCRS,
        pltPatches,
        pltPath,
        seiscmap,
        shp,
        xr,
    )


@app.cell(hide_code=True)
def _(Path):
    filedir = Path(__file__).parent
    filename = filedir / "Kerry3D.segy"
    kerry_url = "http://s3.amazonaws.com/open.source.geoscience/open_data/newzealand/Taranaiki_Basin/Keri_3D/Kerry3D.segy"
    return filedir, filename, kerry_url


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# 3D Seismic Kerry Data Gathering & Preprocessing""")#.center()
    return


@app.cell
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


@app.cell
def _(ccrs, pjCRS):


    LATITUDE_SIGN = "S"
    ESPG = 2193 # New Zealand Transverse Mercator 2000
    PROJ_CRS = pjCRS.from_epsg(ESPG)
    NZTM = ccrs.CRS(PROJ_CRS)
    TRANS_MERC = ccrs.TransverseMercator(central_latitude=-20, central_longitude=180)
    return (ESPG,)


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
def _():
    return


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


@app.cell
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
    return ilines, n_ilines, n_xlines, xlines


@app.cell
def _(n_ilines, n_xlines, nsample, stream):
    sampling_rate = stream[0].stats.delta * 1000
    IL_START = stream[0].stats.segy.trace_header.source_energy_direction_exponent
    XL_START = stream[0].stats.segy.trace_header.ensemble_number
    sample_rate = sampling_rate
    Z_START  = 0 * sample_rate

    IL_END = n_ilines + IL_START
    XL_END = n_xlines + XL_START
    Z_END  = nsample  * sample_rate
    return (
        IL_END,
        IL_START,
        XL_END,
        XL_START,
        Z_END,
        Z_START,
        sample_rate,
        sampling_rate,
    )


@app.cell
def _(np, shp):
    # creating seismic linesstring

    def generate_linestrings(coords    : np.ndarray,
                             ilines_idx: np.ndarray,
                             xlines_idx: np.ndarray) -> tuple[list[shp.LineString, ...]]:
        iline_strings, xline_strings = list(), list()
        _ilines_start = ilines_idx[0]
        _xlines_start = xlines_idx[0]
        _ilines_mapper = map(lambda x: iline_strings.append(shp.LineString(coords[x-_ilines_start, ...])), ilines_idx)
        _xlines_mapper = map(lambda x: xline_strings.append(shp.LineString(coords[:, x-_xlines_start, :])), xlines_idx)
        for _ in _ilines_mapper: pass
        for _ in _xlines_mapper: pass
        return (iline_strings, xline_strings)




    # print(iline_strings, xline_strings)
    return (generate_linestrings,)


@app.cell
def _(seis_coord):
    _a = tuple(tuple([_.tolist() for _ in seis_coord[0, ...]]))
    _b = [tuple(_.tolist()) for _ in seis_coord[0, ...]]

    len(_a[0]), len(_b[0])
    return


@app.cell
def _(Any, plt, xr):

    def get_grid_lines(xr_data : xr.DataArray,
                       nxline  : int,
                       niline  : int,
                       incr_il : int = 20,
                       incr_xl : int = 25,) -> dict[str, Any]:
        _x = list(range(0, nxline+1, incr_xl))
        _i = list(range(0, niline+1, incr_il))
        return dict(
            selection=xr_data.isel(XL=_x, IL=_i),

            xl=_x,
            mid_x=_x[len(_x) // 2],

            il=_i,
            mid_i=_i[len(_i) // 2]
        )

    def make_line_grids(ax: plt.Axes,
                        iline_idxs: list[int], 
                        xline_idxs: list[int],
                        config: dict[str, Any]) -> None:

        _x = config["xl"]
        _i = config["il"]
        _mid_x = config["mid_x"]
        _mid_i = config["mid_i"]

        _n = 10000
        count_idx = 0
        coord_from_xl = [[],[]]
        angle=1.85
        for _idx, _xline in zip(_x, config["selection"]['XLINES'].data):
            _points = _xline.xy
            ax.plot(*_points, "b-", linewidth=0.5, alpha=0.75)
            ax.text(_points[0][0]+(0.01 * _n),
                     _points[1][0]+(0.049 * _n),
                     xline_idxs[_idx], color="blue", fontsize=7.5, rotation=angle)
            if _idx == _mid_x:
                ax.text(_points[0][0]+(0.11 * _n), _points[1][0]+(0.049 * _n),
                        "CROSSLINE", color="blue", fontsize=9.5, rotation=(angle-90))

        for _idx, _iline in zip(_i, config["selection"]['ILINES'].data):
            _points = _iline.xy
            ax.plot(*_points, "r-", linewidth=0.5, alpha=0.75)
            ax.text(_points[0][0]-(0.0450 * _n), _points[1][0]-(0.035 * _n),
                     iline_idxs[_idx], color="red", fontsize=7.5, rotation=(90-angle))
            if _idx == _mid_i:
                ax.text(_points[0][0]-(0.0450 * _n), _points[1][0]-(0.115 * _n),
                        "INLINE", color="red", fontsize=9.5, rotation=angle)

        return None
    return get_grid_lines, make_line_grids


@app.cell
def _(
    IL_START,
    XL_START,
    ilines,
    mo,
    n_ilines,
    n_xlines,
    np,
    nsample,
    ntraces,
    sampling_rate,
    stream,
    xlines,
    xr,
):
    # streaming traces
    Bar_collecting_trace = mo.status.progress_bar(
        title="Collectin Data from Traces",
        completion_title="Data are collected & masked",
        total = ntraces + nsample
    )

    _TWT_IDX = np.arange(0, nsample*sampling_rate, sampling_rate)

    seis_xr = xr.DataArray(
        data=np.zeros((n_ilines, n_xlines, nsample)),
        dims=["IL", "XL", "TWT"],
        coords= dict(
            IL = ilines,
            XL = xlines,
            TWT = _TWT_IDX.astype("float32"),
        ),
    )

    seis_coord = np.zeros((n_ilines, n_xlines, 2))
    seis_lines = np.zeros((n_ilines, n_xlines, 2))

    with Bar_collecting_trace as _bar:
        for i_4 in range(ntraces):
            _bar.update(subtitle=f"Colleting data from trace-{i_4}")
            tracei = stream[i_4]
            il_1 = tracei.stats.segy.trace_header.source_energy_direction_exponent
            xl_1 = tracei.stats.segy.trace_header.ensemble_number
            seis_xr.data[il_1 - IL_START, xl_1 - XL_START] = tracei.data
            seis_coord[il_1 - IL_START][xl_1 - XL_START][0] = tracei.stats.segy.trace_header.source_coordinate_x #/ 10000
            seis_coord[il_1 - IL_START][xl_1 - XL_START][1] = tracei.stats.segy.trace_header.source_coordinate_y #/ -100000
            seis_lines[il_1 - IL_START][xl_1 - XL_START][0] = il_1
            seis_lines[il_1 - IL_START][xl_1 - XL_START][1] = xl_1

        MASK = np.sum(np.abs(seis_xr.data), axis=2)
        MASK = np.where(MASK == 0.000, True, False)
        for _z in range(nsample):
            _bar.update(subtitle=f"Masking data at vertical silce-{_z}")
            seis_xr.data[:, :,_z][MASK] = np.nan


    return MASK, seis_coord, seis_xr


@app.cell
def _(generate_linestrings, ilines, seis_coord, seis_xr, xlines):
    iline_strings, xline_strings = generate_linestrings(coords=seis_coord,
                                                        ilines_idx=ilines,
                                                        xlines_idx=xlines)
    print(len(xlines), len(xline_strings))
    print(len(ilines), len(iline_strings))
    seis_xr.coords.update(
        dict(
            LATITUDE=(["IL", "XL"], seis_coord[:, :, 1]),
            LONGITUDE=(["IL", "XL"], seis_coord[:, :, 0]),
            XLINES = (["XL",], xline_strings),
            ILINES = (["IL",], iline_strings),
        )
    )

    seis_stats = {}
    SEIS_MAX = seis_xr.max().data # np.nanmax(seis_flatten)
    SEIS_MIN = seis_xr.min().data # np.nanmin(seis_flatten)
    SEIS_STD = seis_xr.std().data # np.nanstd(seis_flatten)
    SEIS_MEAN = seis_xr.mean().data # np.nanmean(seis_flatten)
    seis_stats['max'] = SEIS_MAX
    seis_stats['min'] = SEIS_MIN
    seis_stats['std'] = SEIS_STD
    seis_stats['mean'] = SEIS_MEAN
    seis_stats['median'] = seis_xr.median().data # np.nanmedian(seis_flatten)
    seis_stats = [dict(Statistic=stat, Value=val) for stat, val in seis_stats.items()]
    return SEIS_MAX, SEIS_MIN, seis_stats


@app.cell
def _(get_grid_lines, n_ilines, n_xlines, seis_coord, seis_xr):

    grid_lines_configs = get_grid_lines(
        xr_data    = seis_xr,
        niline     = n_ilines, nxline     = n_xlines,
        incr_il    = 20,       incr_xl    = 25,
    )
    scale_bar = dict(
                    dx=1, label="Scale", dimension="si-length",
                    scale_loc="right", label_loc="left", location="lower left",
                    frameon=True, color="#000000", scale_formatter = lambda value, unit: f"{value} {unit}",
                    pad=0.5, box_alpha=0.8,
                    rotation='horizontal-only'
                )
    global_coord_bounds = dict(
        lon_min = seis_coord[..., 1].min(),
        lon_max = seis_coord[..., 1].max(),
        lat_min = seis_coord[..., 0].min(),
        lat_max = seis_coord[..., 0].max(),
    )
    # seis_xr
    return global_coord_bounds, grid_lines_configs, scale_bar


@app.cell
def _(plt):
    plt.style.use("bmh")
    # plt.style.available
    return


@app.cell
def _(BytesIO, namedtuple, plt):


    TpVal = namedtuple("TpVal", ['value'])
    def save_fig_buf(f):
        buf = BytesIO()
        if f == None:
            plt.gcf()
            plt.savefig(buf, bbox_inches='tight', format="png", dpi=300.0)
        else:
            f.savefig(buf, bbox_inches='tight', format="png", dpi=300.0)
        return buf

    def save_tex_buf(string):
        return string.encode("utf-8")
    return TpVal, save_fig_buf, save_tex_buf


@app.cell
def _(mo):
    run_boundary_from = mo.ui.run_button(label="Reset!")
    return (run_boundary_from,)


@app.cell
def _(
    IL_END,
    IL_START,
    XL_END,
    XL_START,
    Z_END,
    Z_START,
    mo,
    run_boundary_from,
    sample_rate,
):
    boundary_form = mo.md("""
    | Boundary  | Start | Stop  |
    | :-------: | :---: | :--:  |
    | CROSSLINE | {xl1} | {xl2} |
    | INLINE    | {il1} | {il2} |
    | DEPTH     | {z1}  | {z2}  |
    """).center().batch(
        xl1 = mo.ui.number(start=XL_START, stop=XL_END, step=1, label="", value=45 + 58),
        xl2 = mo.ui.number(start=XL_START, stop=XL_END, step=1, label="", value=45 + 58 + 608),
        il1 = mo.ui.number(start=IL_START, stop=IL_END, step=1, label="", value=510 + 15,),
        il2 = mo.ui.number(start=IL_START, stop=IL_END, step=1, label="", value=510 + 15 + 192,),
        z1 = mo.ui.number(start=Z_START,  stop=Z_END,  step=sample_rate, label="", value=25*4),
        z2 = mo.ui.number(start=Z_START,  stop=Z_END,  step=sample_rate, label="", value=(272 + 25) * 4),
    ).form()
    if run_boundary_from.value:
        boundary_form = mo.md("""
    | Boundary  | Start | Stop  |
    | :-------: | :---: | :--:  |
    | CROSSLINE | {xl1} | {xl2} |
    | INLINE    | {il1} | {il2} |
    | DEPTH     | {z1}  | {z2}  |
    """).center().batch(
            xl1 = mo.ui.number(start=XL_START, stop=XL_END, step=1, label="", value=45 + 58),
            xl2 = mo.ui.number(start=XL_START, stop=XL_END, step=1, label="", value=45 + 58 + 608),
            il1 = mo.ui.number(start=IL_START, stop=IL_END, step=1, label="", value=510 + 15,),
            il2 = mo.ui.number(start=IL_START, stop=IL_END, step=1, label="", value=510 + 15 + 192,),
            z1 = mo.ui.number(start=Z_START,  stop=Z_END,  step=sample_rate, label="", value=25*4),
            z2 = mo.ui.number(start=Z_START,  stop=Z_END,  step=sample_rate, label="", value=(272 + 25) * 4),
        ).form()
    mo.vstack([run_boundary_from, boundary_form])
    return (boundary_form,)


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
    return vminmax, z_idx, z_idx_2


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
        n           : int | None = None
    ) -> mo.Html:
        fig_iline, ax_iline = plt.subplots(1,2, figsize=(10,7), sharey=False, layout="compressed")
        ax_iline[0].imshow(data[idxs[0], :, :].transpose(), seiscmap(), aspect="auto", **minmax)
        ax_iline[1].imshow(data[idxs[1], :, :].transpose(), seiscmap(), aspect="auto", **minmax)

        _n = 75 if not n else n
        _m = 1000

        _yticks = np.arange(0, dims[2], _n)
        _ytickslabel = [
            str(x)[:5] for x in np.linspace(
                dim_start[2],
                ( (dims[2]*4) + dim_start[2] ),
                len(_yticks),
                endpoint=False)/_m
        ]

        for _ax, _d in zip(ax_iline, buttons):
            _ax.set_ylabel("TWT (s)", fontsize=15)
            _ax.set_xlabel("CROSSLINE", fontsize=15, labelpad=10)

            _ax.set_xticks(
                np.arange(0, dims[1], _n),
                np.arange(dim_start[1], dims[1]+dim_start[1], _n),
                rotation=90)

            _ax.set_yticks(_yticks, _ytickslabel, rotation=0)
            _axR = _ax.secondary_yaxis('right')
            _axR.set_yticks(_yticks, _ytickslabel, rotation=0)

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

    return


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
def _(
    Any,
    ESPG,
    XL_END,
    XL_START,
    ilines,
    mo,
    north_arrow,
    np,
    plt,
    save_fig_buf,
    seiscmap,
    vminmax,
    xlines,
    xr,
):
    def plot_ilines_2_(
        data: xr.DataArray,
        axis        : str,
        buttons     : tuple[int, int],
        vertbuttons : tuple[int, int],
        horbuttons  : tuple[int, int],
        title       : str = "KerryOriginal",
        minmax      : dict[str, float|int] = vminmax,
        iline_idxs  : list[int] = ilines,
        xline_idxs  : list[int] = xlines,
        projection  : int = ESPG,
        scale_bar   : dict[str, Any] | None = None,
        line_bounds : dict[str, np.ndarray] = dict(
            zmin = 0,
            zmax = 5000,
            xmin = XL_START,
            xmax = XL_END
        ),
    ) -> mo.Html:

        _fig, _axs = plt.subplots(1, 2, figsize=(10,7), 
                                  sharey=True,
                                  layout="compressed", dpi=150)

        _labels = dict(ylabel="TWT (ms)", xlabel="CROSSLINE")

        _plotting_vars = dict(y="TWT", x="XL", cmap=seiscmap(), ) 
        _plotting_cbar = dict(shrink=0.6, format="%2.1f", label="Seismic Amplitude",
                              ticks=np.linspace(minmax["vmin"], minmax["vmax"], 10, endpoint=True))

        _offset_limx = 0; _offset_limy = 0
        _count = 0
        for _ax, _twt_idx in zip(_axs, buttons):
            _ax.set(aspect="auto")

            # # _data = data.isel({axis:_twt_idx})
            # print(data)
            # print(data.sel({axis:_twt_idx}))

            if _count != 0: # last subplot
                _labels.update({"ylabel":""})
                north_arrow(ax=_ax, location="upper right", rotation={"degrees":88})
                data.sel({axis:_twt_idx}).plot(
                    ax=_ax, **_plotting_vars, add_colorbar=True, cbar_kwargs=_plotting_cbar, **minmax)

            else:
                data.sel({axis:_twt_idx}).plot(
                    ax=_ax, **_plotting_vars, **minmax, add_colorbar=False)

            _ax.set_title(f"${axis}\ {_twt_idx}$",
                          fontsize=20, fontweight="bold",
                          loc="center", y=1.025)

            _ymin, _ymax, _xmin, _xmax = list(line_bounds.values())
            _yticks = np.linspace(_ymin, _ymax, 10)
            _ax.set_yticks(_yticks, [f"{x:3.2f}" for x in _yticks], rotation=0)
            _ax.grid(which="both", color="black", alpha=0.75, markevery=2, snap=True)

            _ax.set_xlim(_xmin - _offset_limx, _xmax + _offset_limx)
            _ax.set_ylim(_ymin - _offset_limy, _ymax + _offset_limy)
            _ax.invert_yaxis()

            _data2 = data.sel(XL=list(vertbuttons), TWT=list(horbuttons), method="nearest")
            for _idx_z, _line_z, _line_styl in zip(vertbuttons, _data2["XL"].data, ("-", "--")):
                _ax.axvline(_line_z, linewidth=2, linestyle=_line_styl, color="blue", label=f"CROSSLINE {_idx_z}")
            for _idx_il, _line_il, _line_styl in zip(horbuttons, _data2["TWT"].data, ("-", "--")):
                _ax.axhline(_line_il, linewidth=2, linestyle=_line_styl, color="black", label=f"TWT {_idx_il} ms")

            # if _count == 0 and scale_bar: # last subplot
            #     _ax.add_artist(ScaleBar(**scale_bar))

            _ax.set(**_labels)
            _count += 1

        _handles, _labels = _axs[1].get_legend_handles_labels()
        _fig.legend(_handles, _labels, ncols=4, loc='lower center', bbox_to_anchor =(0.5,-0.05))
        _fig_file_name = [
            title,
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

    return (plot_ilines_2_,)


@app.cell(hide_code=True)
def _():
    # plot_ilines_2_(
    #     data        = seis_xr.sel(IL=[inline_full_num.value, inline_full_num_2.value]),
    #     axis        = "IL",
    #     buttons     = (inline_full_num.value, inline_full_num_2.value),
    #     vertbuttons = (xline_full_num.value,  xline_full_num_2.value),
    #     horbuttons  = (depth_full_num.value,  depth_full_num_2.value),
    #     title       = "KerryOriginal",
    #     minmax      = vminmax,
    #     scale_bar   = scale_bar,
    # )
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
        minmax      : dict[str, float] = vminmax,
        n           : int | None = None
    ) -> mo.Html:

        fig_xline, ax_xline = plt.subplots(1,2, figsize=(10,7), sharey=False, layout="compressed")
        _n = 75 if not n else n
        _m = 1000

        ax_xline[0].imshow(data[:,idxs[0],:].transpose(), seiscmap(), aspect="auto", **minmax)
        ax_xline[1].imshow(data[:,idxs[1],:].transpose(), seiscmap(), aspect="auto", **minmax)

        _yticks = np.arange(0, dims[2], _n)
        _ytickslabel = [
            str(x)[:5] for x in np.linspace(
                dim_start[2],
                ( (dims[2]*4) + dim_start[2] ),
                len(_yticks),
                endpoint=False)/_m
        ]

        for _ax, _d in zip(ax_xline, buttons):
            _ax.set_ylabel("TWT (s)", fontsize=15)
            _ax.set_xlabel("INLINE", labelpad=10, fontsize=15)

            _ax.set_yticks(_yticks, _ytickslabel, rotation=0)
            _axR = _ax.secondary_yaxis('right')
            _axR.set_yticks(_yticks, _ytickslabel, rotation=0)

            _ax.set_title(f"$CROSSLINE\ {_d}$", fontsize=20, style="italic", pad=10)
            _ax.axvline(vertidxs[0], color="red", label=f"INLINE {vertbuttons[0]}")
            _ax.axvline(vertidxs[1], color="red", linestyle="--", label=f"INLINE {vertbuttons[1]}")
            _ax.axhline(horidxs[0], color="black", label=f"TWT {horbuttons[0]} ms")
            _ax.axhline(horidxs[1], color="black", linestyle="--", label=f"TWT {horbuttons[1]} ms")
            _ax.grid(which="both", color="black", alpha=0.25, markevery=2, snap=True)

        _handles, _labels = ax_xline[1].get_legend_handles_labels()
        fig_xline.legend(_handles, _labels, ncols=4, loc='lower center', bbox_to_anchor =(0.5,-0.05))

        _fig_file_name = [title, 
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

    return


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
def _(
    Any,
    ESPG,
    IL_END,
    IL_START,
    ilines,
    mo,
    north_arrow,
    np,
    plt,
    save_fig_buf,
    seiscmap,
    vminmax,
    xlines,
    xr,
):
    def plot_xlines_2_(
        data: xr.DataArray,
        axis        : str,
        buttons     : tuple[int, int],
        vertbuttons : tuple[int, int],
        horbuttons  : tuple[int, int],
        title       : str = "KerryOriginal",
        minmax      : dict[str, float|int] = vminmax,
        iline_idxs  : list[int] = ilines,
        xline_idxs  : list[int] = xlines,
        projection  : int = ESPG,
        scale_bar   : dict[str, Any] | None = None,
        line_bounds : dict[str, np.ndarray] = dict(
            zmin = 0,
            zmax = 5000,
            xmin = IL_START,
            xmax = IL_END
        ),
    ) -> mo.Html:

        _fig, _axs = plt.subplots(1, 2, figsize=(13,11), 
                                  sharey=True,
                                  layout="compressed", dpi=150)

        _labels = dict(ylabel="TWT (ms)", xlabel="INLINE")

        _plotting_vars = dict(y="TWT", x="IL", cmap=seiscmap(), ) 
        _plotting_cbar = dict(shrink=0.6, format="%2.1f", label="Seismic Amplitude",
                              ticks=np.linspace(minmax["vmin"], minmax["vmax"], 10, endpoint=True))

        _offset_limx = 0; _offset_limy = 0
        _count = 0
        for _ax, _twt_idx in zip(_axs, buttons):
            _ax.set(aspect="auto")

            if _count != 0: # last subplot
                _labels.update({"ylabel":""})
                north_arrow(ax=_ax, location="upper right", rotation={"degrees":-2})
                data.sel({axis:_twt_idx}).plot(
                    ax=_ax, **_plotting_vars, add_colorbar=True, cbar_kwargs=_plotting_cbar, **minmax)

            else:
                data.sel({axis:_twt_idx}).plot(
                    ax=_ax, **_plotting_vars, **minmax, add_colorbar=False)

            _ax.set_title(f"${axis}\ {_twt_idx}$",
                          fontsize=20, fontweight="bold",
                          loc="center", y=1.025)

            _ymin, _ymax, _xmin, _xmax = list(line_bounds.values())
            _yticks = np.linspace(_ymin, _ymax, 10)
            _ax.set_yticks(_yticks, [f"{x:3.2f}" for x in _yticks], rotation=0)
            _ax.grid(which="both", color="black", alpha=0.75, markevery=2, snap=True)

            _ax.set_xlim(_xmin - _offset_limx, _xmax + _offset_limx)
            _ax.set_ylim(_ymin - _offset_limy, _ymax + _offset_limy)
            _ax.invert_yaxis()

            _data2 = data.sel(IL=list(vertbuttons), TWT=list(horbuttons), method="nearest")
            for _idx_z, _line_z, _line_styl in zip(vertbuttons, _data2["IL"].data, ("-", "--")):
                _ax.axvline(_line_z, linewidth=2, linestyle=_line_styl, color="red", label=f"INLINE {_idx_z}")
            for _idx_il, _line_il, _line_styl in zip(horbuttons, _data2["TWT"].data, ("-", "--")):
                _ax.axhline(_line_il, linewidth=2, linestyle=_line_styl, color="black", label=f"TWT {_idx_il} ms")

            # if _count == 0 and scale_bar: # last subplot
            #     _ax.add_artist(ScaleBar(**scale_bar))

            _ax.set(**_labels)
            _count += 1

        _handles, _labels = _axs[1].get_legend_handles_labels()
        _fig.legend(_handles, _labels, ncols=4, loc='lower center', bbox_to_anchor =(0.5,-0.05))
        _fig_file_name = [title, 
                          "XL", str(buttons[0])     + "-" + str(buttons[1]),
                          "IL", str(vertbuttons[0]) + "-" + str(vertbuttons[1]),
                          "TWT", str(horbuttons[0]) + "-" + str(horbuttons[1])
                         ]
        _fig_file_name = "_".join(_fig_file_name)

        _download_lazy = mo.download(
            data = save_fig_buf(plt.gcf()),
            filename = _fig_file_name,
            label = _fig_file_name
        )
        return mo.vstack([mo.as_html(plt.gcf()).center(), _download_lazy.center()]).center()

    return (plot_xlines_2_,)


@app.cell
def _():
    # plot_xlines_2_(
    #     data        = seis_xr.sel(XL=[xline_full_num.value, xline_full_num_2.value]),
    #     axis        = "XL",
    #     buttons     = (xline_full_num.value,  xline_full_num_2.value),
    #     vertbuttons = (inline_full_num.value, inline_full_num_2.value),
    #     horbuttons  = (depth_full_num.value,  depth_full_num_2.value),
    #     title       = "KerryOriginal",
    #     minmax      = vminmax,
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
        n           : int | None = None
    ) -> mo.Html:
        fig_depth, ax_depth = plt.subplots(1,2, figsize=(10,10), sharey=False, layout="compressed")
        ax_depth[0].imshow(data[:,:,idxs[0]].T, seiscmap(), aspect="equal", origin='upper', **minmax)
        ax_depth[1].imshow(data[:,:,idxs[1]].T, seiscmap(), aspect="equal", origin='upper', **minmax)
        _n = 25 if not n else n
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
    return


@app.cell
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
def _(
    Any,
    ESPG,
    ScaleBar,
    global_coord_bounds,
    grid_lines_configs,
    ilines,
    make_line_grids,
    mo,
    north_arrow,
    np,
    plt,
    sampling_rate,
    save_fig_buf,
    seiscmap,
    vminmax,
    xlines,
    xr,
):
    def plot_timedepths_2_(
        data: xr.DataArray,
        grid_lines_config : dict[str, Any],
        axis        : str,
        idxs        : tuple[int, int],
        buttons     : tuple[int, int],
        vertbuttons : tuple[int, int],
        horbuttons  : tuple[int, int],
        title       : str = "KerryOriginal",
        minmax      : dict[str, float|int] = vminmax,
        iline_idxs  : list[int] = ilines,
        xline_idxs  : list[int] = xlines,
        projection  : int = ESPG,
        scale_bar   : dict[str, Any] | None = None,
        line_bounds : dict[str, np.ndarray] = global_coord_bounds,
    ) -> mo.Html:

        _fig, _axs = plt.subplots(1, 2, figsize=(13,11), 
                                  sharey=True,
                                  layout="compressed", dpi=150)

        _labels = dict(ylabel="NORTHING", xlabel="EASTING")

        _plotting_vars = dict(y="LATITUDE", x="LONGITUDE", cmap=seiscmap(), ) 
        _plotting_cbar = dict(shrink=0.6, format="%2.1f", label="Seismic Amplitude",
                              ticks=np.linspace(minmax["vmin"], minmax["vmax"], 10, endpoint=True))

        _offset_limx = 2250; _offset_limy = 2000
        _count = 0
        for _ax, _twt_idx in zip(_axs, idxs):
            _twt_idx *= sampling_rate
            _ax.set(aspect="auto")

            _data = data.sel({axis:_twt_idx})

            if _count != 0: # last subplot
                _labels.update({"ylabel":""})
                north_arrow(ax=_ax, location="upper right", rotation={"degrees":0})
                _data.plot(ax=_ax, **_plotting_vars, add_colorbar=True, cbar_kwargs=_plotting_cbar, **minmax)

            else:
                _data.plot(ax=_ax, **_plotting_vars, **minmax, add_colorbar=False)

            _ax.set_title(f"${axis}\ {_twt_idx}\ ms$",
                          fontsize=20, fontweight="bold",
                          loc="center", y=1.025)

            make_line_grids(ax         = _ax,
                            iline_idxs = iline_idxs,
                            xline_idxs = xline_idxs,
                            config     = grid_lines_configs)

            _data2 = _data.sel(XL=list(vertbuttons), IL=list(horbuttons), method="nearest")
            for _idx_xl, _line_xl, _line_styl in zip(vertbuttons, _data2["XLINES"].data, ("-", "--")):
                _ax.plot(*_line_xl.xy, linewidth=2, linestyle=_line_styl,
                         color="blue", label=f"CROSSLINE {_idx_xl}")
            for _idx_il, _line_il, _line_styl in zip(horbuttons, _data2["ILINES"].data, ("-", "--")):
                _ax.plot(*_line_il.xy, linewidth=2, linestyle=_line_styl,
                         color="red", label=f"INLINE {_idx_il}")

            if _count == 0 and scale_bar: # last subplot
                _ax.add_artist(ScaleBar(**scale_bar))

            _ymin, _ymax, _xmin, _xmax = list(global_coord_bounds.values())
            _yticks = np.linspace(_ymin, _ymax, 10)
            _ax.set_yticks(_yticks, [f"{x:1.2E}" for x in _yticks], rotation=90)
            _ax.grid(which="both", color="black", alpha=0.45, markevery=2, snap=True)

            _ax.set_xlim(_xmin - _offset_limx, _xmax + _offset_limx)
            _ax.set_ylim(_ymin - _offset_limy, _ymax + _offset_limy)
            _ax.invert_yaxis()

            _ax.set(**_labels)
            _count += 1

        _handles, _labels = _axs[1].get_legend_handles_labels()
        _fig.legend(_handles, _labels, ncols=4, loc='lower center', bbox_to_anchor =(0.5,-0.05))
        _fig_file_name = [title,
                          "TWT", str(buttons[0])     + "-" + str(buttons[1]),
                          "XL", str(vertbuttons[0]) + "-" + str(vertbuttons[1]),
                          "IL", str(horbuttons[0]) + "-" + str(horbuttons[1])
                         ]
        _fig_file_name = "_".join(_fig_file_name)

        _download_lazy = mo.download(
            data = save_fig_buf(plt.gcf()),
            filename = _fig_file_name,
            label = _fig_file_name
        )
        return mo.vstack([mo.as_html(plt.gcf()).center(), _download_lazy.center()]).center()

    return (plot_timedepths_2_,)


@app.cell(hide_code=True)
def _():
    # plot_timedepths_2_(
    #     data        = seis_xr,
    #     grid_lines_config = grid_lines_configs,
    #     axis        = "TWT",
    #     idxs        = (z_idx, z_idx_2),
    #     buttons     = (depth_full_num.value,  depth_full_num_2.value),
    #     vertbuttons = (xline_full_num.value,  xline_full_num_2.value),
    #     horbuttons  = (inline_full_num.value, inline_full_num_2.value),
    #     title       = "KerryOriginal",
    #     minmax      = vminmax,
    #     scale_bar   = scale_bar,
    # )
    return


@app.cell
def _(MASK, filedir, mo, seis_xr, time):
    def save_on_click(data=seis_xr, suffix="", save_mask=False, mask=MASK):
        t0_4 = time.time()
        with mo.status.progress_bar(
            total=2 if save_mask else 1,
            title="Saving The Data",
            subtitle="Please wait",
        ) as _bar:
            _bar.update(subtitle="Saving Data")
            _x = seis_xr.reset_coords(names=["XLINES", "ILINES"], drop=True)
            _x.to_netcdf(filedir / f"kerry3d{suffix}.nc")
            if save_mask:
                # np.savez(filedir / f"kerry3d_mask{suffix}", mask.astype(np.int16))
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
        mo.ui.text_area(_latex_tabel, rows=10, full_width=True, max_length=1000),
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
def _(hist_binsize, np, pltPatches, pltPath):
    def make_histogram_seis(data: np.ndarray, in_bins=hist_binsize.value):
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
def _(make_histogram_seis, np, plt, seis_xr):
    fig_stat = plt.figure(figsize=(5,5))
    ax_stat = fig_stat.add_subplot(1,1, 1)
    N, bins, patches = make_histogram_seis(seis_xr.data[~np.isnan(seis_xr.data)])
    # FIG_STAT = mo.download(
    #     data = save_fig_buf(f=None),
    #     filename = _fig_file_name,
    #     label = _fig_file_name
    # )

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
    plt,
    save_fig_buf,
    table_stat,
):
    line_colors_stats = cycle(["r", 'k', 'b', 'g', 'orange'])

    n_stats_to_plot = len(table_stat.value)
    ax_stat.clear()
    ax_stat.add_patch(patches)
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
    _ = plt.gcf()
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
def _(masked_crop_data):
    SEIS_MAX_stdr = masked_crop_data.max().data # np.nanmax(seis_flatten_stdr)
    SEIS_MIN_stdr = masked_crop_data.min().data # np.nanmin(seis_flatten_stdr)
    seis_stats_stdr = {}
    seis_stats_stdr['max'] = SEIS_MAX_stdr
    seis_stats_stdr['min'] = SEIS_MIN_stdr
    seis_stats_stdr['std'] = masked_crop_data.std().data
    seis_stats_stdr['mean'] = masked_crop_data.mean().data # (seis_flatten_stdr)
    seis_stats_stdr['median'] = masked_crop_data.median().data # (seis_flatten_stdr)
    seis_stats_stdr = [dict(Statistic=stat, Value=val) for stat, val in seis_stats_stdr.items()]
    return (seis_stats_stdr,)


@app.cell
def _(mo, pd, save_tex_buf, seis_stats_stdr):
    _df = pd.DataFrame(seis_stats_stdr)
    _filename = "KerryStandardize_hist"
    _latex_tabel = f"""\\begin{{tabel}}[ht!]
    \\caption{{{_filename}}}
    \\label{{tab:{_filename}}}
    {_df.to_latex(index=False)}\\end{{tabel}}"""
    _download_lazy = mo.download(
        data = save_tex_buf(_latex_tabel), filename = _filename + ".tex",
        label=_filename + ".tex")

    std_latex = [
        mo.hstack([
            mo.md(r"""## Cropped & Standardization""").center(),
            _download_lazy.center()
        ]),
        mo.ui.text_area(_latex_tabel,
                        rows=10,
                        full_width=True,
                        max_length=1000),
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
def _(make_histogram_seis, masked_crop_data, np, plt):
    fig_stat_stdr, ax_stat_stdr = plt.subplots(1,1, figsize=(5,5))
    N_stdr, bins_stdr, patches_stdr = make_histogram_seis(
        data=masked_crop_data.data[~np.isnan(masked_crop_data.data)])
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
def _():
    # template_zero = np.zeros_like(seis_np)
    return


@app.cell
def _(
    IL_START,
    XL_START,
    depth_crop_end,
    depth_crop_start,
    il_crop_end,
    il_crop_start,
    sample_rate,
    seis_xr,
    xl_crop_end,
    xl_crop_start,
):
    # normalized_seis_np = ((seis_np - SEIS_MIN) / (SEIS_MAX - SEIS_MIN))
    # normalized_seis_np = 2 * ((normalized_seis_np - 0) / (1 - 0)) - 1

    # normalized_seis_np = 2 * ((seis_np - SEIS_MIN) / (SEIS_MAX - SEIS_MIN)) - 1

    # normalized_seis_np = (seis_np - SEIS_MEAN) / SEIS_STD

    # normalized_seis_np = seis_np

    # masked_crop_data = template_zero * np.nan
    # # masked_crop_data *= np.nan
    # masked_crop_data[
    #         slice(il_crop_start - IL_START, il_crop_end - IL_START),
    #         slice(xl_crop_start - XL_START, xl_crop_end - XL_START),
    #         slice(depth_crop_start//sample_rate, depth_crop_end//sample_rate)] = normalized_seis_np[
    #             slice(il_crop_start - IL_START, il_crop_end - IL_START),
    #             slice(xl_crop_start - XL_START, xl_crop_end - XL_START),
    #             slice(depth_crop_start//sample_rate, depth_crop_end//sample_rate)
    # ]
    masked_crop_data = seis_xr.isel(
        IL=slice(il_crop_start - IL_START, il_crop_end - IL_START),
        XL=slice(xl_crop_start - XL_START, xl_crop_end - XL_START),
        TWT=slice(
            int(depth_crop_start//sample_rate),
            int(depth_crop_end//sample_rate)
        ))
    # masked_crop_data
    return (masked_crop_data,)


@app.cell(hide_code=True)
def _(
    depth_full_num,
    depth_full_num_2,
    inline_full_num,
    inline_full_num_2,
    masked_crop_data,
    mo,
    plot_ilines_2_,
    scale_bar,
    seis_xr,
    vminmax,
    xline_full_num,
    xline_full_num_2,
):
    mo.hstack([
        mo.vstack([
            mo.md("## Original").center(),
            plot_ilines_2_(
                data        = seis_xr.sel(IL=[inline_full_num.value, inline_full_num_2.value]),
                axis        = "IL",
                buttons     = (inline_full_num.value, inline_full_num_2.value),
                vertbuttons = (xline_full_num.value,  xline_full_num_2.value),
                horbuttons  = (depth_full_num.value,  depth_full_num_2.value),
                title       = "KerryOriginal",
                minmax      = vminmax,
                scale_bar   = scale_bar,
            ),
        ]),
        mo.vstack([
            mo.md("## Masked").center(),
            plot_ilines_2_(
                data        = masked_crop_data.sel(IL=[inline_full_num.value, inline_full_num_2.value-1]),
                axis        = "IL",
                buttons     = (inline_full_num.value, inline_full_num_2.value-1),
                vertbuttons = (xline_full_num.value,  xline_full_num_2.value-1),
                horbuttons  = (depth_full_num.value,  depth_full_num_2.value-4),
                title       = "KerryOriginal",
                minmax      = vminmax,
                scale_bar   = scale_bar,
            ),
        ])
    ])
    return


@app.cell
def _(
    depth_full_num,
    depth_full_num_2,
    inline_full_num,
    inline_full_num_2,
    masked_crop_data,
    mo,
    plot_xlines_2_,
    seis_xr,
    vminmax,
    xline_full_num,
    xline_full_num_2,
):
    mo.hstack([
        mo.vstack([
            mo.md("## Original").center(),
            plot_xlines_2_(
                data        = seis_xr.sel(XL=[xline_full_num.value, xline_full_num_2.value]),
                axis        = "XL",
                buttons     = (xline_full_num.value,  xline_full_num_2.value),
                vertbuttons = (inline_full_num.value, inline_full_num_2.value),
                horbuttons  = (depth_full_num.value,  depth_full_num_2.value),
                title       = "KerryOriginal",
                minmax      = vminmax,
            )]),
        mo.vstack([
            mo.md("## Masked").center(),
            plot_xlines_2_(
                data        = masked_crop_data.sel(XL=[xline_full_num.value, xline_full_num_2.value-1]),
                axis        = "XL",
                buttons     = (xline_full_num.value,  xline_full_num_2.value-1),
                vertbuttons = (inline_full_num.value, inline_full_num_2.value-1),
                horbuttons  = (depth_full_num.value,  depth_full_num_2.value-4),
                title       = "KerryOriginal",
                minmax      = vminmax,
            )
        ])
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


@app.cell
def _(
    depth_full_num,
    depth_full_num_2,
    grid_lines_configs,
    inline_full_num,
    inline_full_num_2,
    masked_crop_data,
    mo,
    plot_timedepths_2_,
    seis_xr,
    vminmax,
    xline_full_num,
    xline_full_num_2,
    z_idx,
    z_idx_2,
):
    mo.hstack([
        mo.vstack([
                mo.md("## Original").center(),
                plot_timedepths_2_(
                    data        = seis_xr,
                    grid_lines_config = grid_lines_configs,
                    axis        = "TWT",
                    idxs        = (z_idx, z_idx_2),
                    buttons     = (depth_full_num.value,  depth_full_num_2.value),
                    vertbuttons = (xline_full_num.value,  xline_full_num_2.value),
                    horbuttons  = (inline_full_num.value, inline_full_num_2.value),
                    title       = "KerryOriginal",
                    minmax      = vminmax,
                )
        ]),
        mo.vstack([
            mo.md("## Masked").center(),
            plot_timedepths_2_(
                data        = masked_crop_data,
                grid_lines_config = grid_lines_configs,
                axis        = "TWT",
                idxs        = (z_idx, z_idx_2-1),
                buttons     = (depth_full_num.value-1,  depth_full_num_2.value-1),
                vertbuttons = (xline_full_num.value-1,  xline_full_num_2.value-1),
                horbuttons  = (inline_full_num.value, inline_full_num_2.value-4),
                title = "KerryMasked", 
                # minmax = {"vmin":-1, "vmax":1},
            )])
    ])
    return


@app.cell(hide_code=True)
def _(masked_crop_data, np):
    mask_tb_cropped = masked_crop_data.data.astype(np.bool_)

    # print(to_be_saved_cropped.shape, mask_tb_cropped.shape)
    return


@app.cell
def _(masked_crop_data):
    SEIS_CROP_MEAN = masked_crop_data.mean().data
    SEIS_CROP_STD = masked_crop_data.std().data
    SEIS_MAX_stdr_crop = masked_crop_data.max().data
    SEIS_MIN_stdr_crop = masked_crop_data.min().data
    seis_stats_stdr_crop = {}
    seis_stats_stdr_crop['max'] = SEIS_MAX_stdr_crop
    seis_stats_stdr_crop['min'] = SEIS_MIN_stdr_crop
    seis_stats_stdr_crop['std'] = SEIS_CROP_STD
    seis_stats_stdr_crop['mean'] = SEIS_CROP_MEAN
    seis_stats_stdr_crop['median'] = masked_crop_data.median().data
    seis_stats_stdr_crop = [dict(Statistic=stat, Value=val) for stat, val in seis_stats_stdr_crop.items()]
    return (seis_stats_stdr_crop,)


@app.cell
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
        mo.ui.text_area(_latex_tabel, rows=10, full_width=True, max_length=1000),
    ]
    return (croped_latex,)


@app.cell
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
def _(make_histogram_seis, masked_crop_data, np, plt):
    fig_stat_crop, ax_stat_crop = plt.subplots(1,1, figsize=(5,5))
    N_crop, bins_crop, patches_crop = make_histogram_seis(
        data=masked_crop_data.data[~np.isnan(masked_crop_data.data)])
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


@app.cell
def _(
    depth_crop_end,
    depth_crop_start,
    il_crop_end,
    il_crop_start,
    mo,
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
        BUTTON_CROPPED = save_on_click(data=to_be_saved_cropped, suffix=_suffix, save_mask=False, mask=None)
    else:
        BUTTON_CROPPED = mo.md("### Save Cropped data to `.npz`").center()
    return (BUTTON_CROPPED,)


@app.cell
def _():
    return


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
    return


@app.cell
def _(masked_crop_data):
    cropped_shape = masked_crop_data.shape
    return (cropped_shape,)


@app.cell
def _(cropped_shape):
    cropped_shape
    return


@app.cell
def _(
    depth_full_crp,
    depth_full_crp_2,
    inline_full_crp,
    inline_full_crp_2,
    masked_crop_data,
    mo,
    plot_ilines_2_,
    vminmax,
    xline_full_crp,
    xline_full_crp_2,
):
    mo.vstack([
        mo.md("## Cropped").center(),
        plot_ilines_2_(
                data        = masked_crop_data.sel(IL=[inline_full_crp.value, inline_full_crp_2.value-1]),
                axis        = "IL",
                buttons     = (inline_full_crp.value, inline_full_crp_2.value-1),
                vertbuttons = (xline_full_crp.value,  xline_full_crp_2.value-1),
                horbuttons  = (depth_full_crp.value,  depth_full_crp_2.value-4),
                title       = "KerryOriginal",
                minmax      = vminmax,
                line_bounds = dict(
                    zmin = depth_full_crp.value,
                    zmax = depth_full_crp_2.value-4,
                    xmin = xline_full_crp.value,
                    xmax = xline_full_crp_2.value-1
                ),
            ),
    ])
    return


@app.cell
def _(
    depth_full_crp,
    depth_full_crp_2,
    depth_full_num,
    depth_full_num_2,
    inline_full_crp,
    inline_full_crp_2,
    inline_full_num,
    inline_full_num_2,
    masked_crop_data,
    mo,
    plot_xlines_2_,
    vminmax,
    xline_full_num,
    xline_full_num_2,
):
    mo.vstack([
        mo.md("## Cropped").center(),
        plot_xlines_2_(
                data        = masked_crop_data.sel(XL=[xline_full_num.value, xline_full_num_2.value-1]),
                axis        = "XL",
                buttons     = (xline_full_num.value,  xline_full_num_2.value-1),
                vertbuttons = (inline_full_num.value, inline_full_num_2.value-1),
                horbuttons  = (depth_full_num.value,  depth_full_num_2.value-4),
                title       = "KerryOriginal",
                minmax      = vminmax,
                line_bounds = dict(
                    zmin = depth_full_crp.value,
                    zmax = depth_full_crp_2.value-4,
                    xmin = inline_full_crp.value,
                    xmax = inline_full_crp_2.value-1
                ),
            )
    ])
    return


@app.cell
def _(
    depth_full_crp,
    depth_full_crp_2,
    grid_lines_configs,
    inline_full_crp,
    inline_full_crp_2,
    masked_crop_data,
    mo,
    plot_timedepths_2_,
    xline_full_crp,
    xline_full_crp_2,
    z_idx,
    z_idx_2,
):
    mo.vstack([
        mo.md("## Cropped").center(),
        plot_timedepths_2_(
                data        = masked_crop_data,
                grid_lines_config = grid_lines_configs,
                axis        = "TWT",
                idxs        = (z_idx, z_idx_2-1),
                buttons     = (depth_full_crp.value,  depth_full_crp_2.value),
                horbuttons  = (inline_full_crp.value, inline_full_crp_2.value),
                vertbuttons = (xline_full_crp.value,  xline_full_crp_2.value),
                title = "KerryMasked", 
            )
    ])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
