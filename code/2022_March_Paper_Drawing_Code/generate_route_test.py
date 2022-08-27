# %%
import numpy as np

from generate_route import *


stage_route = generate_specifc_dim_route_default(2000, (40, 50))
print(stage_route.shape)

# stage_route = np.load(
#     "/public/home/ssct005t/project/wml_istbi/code/route_default_2dim_100_100.npy", allow_pickle=True)

# stage_route = np.load(
#     "/public/home/ssct005t/project/wml_istbi/code/generate_default_route/route_table/route_default_2dim_100_200.npy",
#     allow_pickle=True)

print(stage_route.shape)
save_route_json(stage_route, "route_table" + '/' + "route_default_2_dim_40_50.json")

confirm_route_table(stage_route)
