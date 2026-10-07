#!/bin/bash
R=/raid/user_juliadollis/julia_docker
for f in \
  $R/runs_riemann/B3_gauss_metrica_teto50/seed_0/best.pt \
  $R/runs_riemann/B3_gauss_metrica_teto50/seed_1/best.pt \
  $R/runs_riemann/B3_gauss_metrica_teto50/seed_2/best.pt \
  $R/runs_riemann/B3_gauss_metrica_teto50/seed_3/best.pt \
  $R/runs_riemann/B3_gauss_metrica_teto50/seed_4/best.pt \
  $R/runs_riemann/B3_gauss_metrica_teto50/seed_5/best.pt \
  $R/runs_riemann/B1_gauss_metrica_teto5/seed_3/best.pt \
  $R/runs_riemann/B0_berhu_size768/seed_0/best.pt \
  $R/runs_riemann/B0_berhu_size768/seed_1/best.pt \
  $R/runs_riemann/B0_berhu_size768/seed_2/best.pt \
  $R/retreinar-deblur/outputs/deblur_docker_4gpu/deblur/checkpoints/best.pt \
  $R/retreinar-deblur/outputs/deblur_docker_4gpu/deblur/deblur.safetensors \
  $R/retreinar-deblur/outputs/deblur_docker_4gpu/deblur/deblur_best_step15500.safetensors \
  $R/genrefocus_deblurnet_paper/outputs/bokehnet_synth_2gpu/bokeh/bokeh.safetensors \
  $R/genrefocus_deblurnet_paper/outputs/bokehnet-geo-B/bokeh/bokeh.safetensors \
  $R/genrefocus_deblurnet_paper/outputs/bokehnet-geo-B/smoke.safetensors \
  $R/genrefocus_deblurnet_paper/outputs/bokehnet-geo-Alinha/bokeh/bokeh.safetensors \
  $R/genrefocus_deblurnet_paper/outputs/bokehnet_fase2_rotac_only/bokeh/bokeh.safetensors \
  $R/genrefocus_deblurnet_paper/outputs/bokehnet_fase2_nofilter/bokeh/.upload_bokeh_step35000.safetensors \
  $R/genrefocus_deblurnet_paper/outputs/bokehnet_fase2_kfix/bokeh/bokeh.safetensors \
  $R/genrefocus_deblurnet_paper/outputs/bokehnet_fase2_4gpu/bokeh/bokeh.safetensors ; do
  sha256sum "$f"
done
