A = {
            "0":{
                "0":"0"
            },
            "subescher start":{
                "subescher uv start":"uv",
                "subescher vw start":"vw",
                "subescher uw start":"uw",
                "subescher uv_w start":"uv_w",
                "subescher uw_v start":"uw_v",
                "subescher vw_u start":"vw_u",
            },
            "subescher end":{
                "subescher uv end":"uv",
                "subescher vw end":"vw",
                "subescher uw end":"uw",
                "subescher uv_w end":"uv_w",
                "subescher uw_v end":"uw_v",
                "subescher vw_u end":"vw_u",
            },
            "1. insertion":{
                "uv 1. insert":"uv",
                "vw 1. insert":"vw",
                "uw 1. insert":"uw",
                "uv_w 1. insert":"uv_w",
                "uw_v 1. insert":"uw_v",
                "vw_u 1. insert":"vw_u",
            }
        }

labels = ["subescher uv start", "subescher uv end", "uv 1. insert", "subescher vw start", "subescher vw end", "vw 1. insert", 
                "subescher uw start", "subescher uw end", "uw 1. insert", "subescher uz start", "subescher uz end", "uz 1. insert",
                "subescher vz start", "subescher vz end", "vz 1. insert", "subescher wz start", "subescher wz end", "wz 1. insert",
                "subescher uv_w start", "subescher uv_w end", "uv_w 1. insert", 
                "subescher uw_v start", "subescher uw_v end", "uw_v 1. insert",
                "subescher vw_u start", "subescher vw_u end", "vw_u 1. insert", 
                "subescher uv_z start", "subescher uv_z end", "uv_z 1. insert",
                "subescher uz_v start", "subescher uz_v end", "uz_v 1. insert",
                "subescher vz_u start", "subescher vz_u end", "vz_u 1. insert",
                "subescher uw_z start", "subescher uw_z end", "uw_z 1. insert", 
                "subescher uz_w start", "subescher uz_w end", "uz_w 1. insert",
                "subescher wz_u start", "subescher wz_u end", "wz_u 1. insert",
                "subescher vw_z start", "subescher vw_z end", "vw_z 1. insert",
                "subescher vz_w start", "subescher vz_w end", "vz_w 1. insert",
                "subescher wz_v start", "subescher wz_v end", "wz_v 1. insert",
                "subescher uvw_z start", "subescher uvw_z end", "uvw_z 1. insert", 
                "subescher uvz_w start", "subescher uvz_w end", "uvz_w 1. insert",
                "subescher uwz_v start", "subescher uwz_v end", "uwz_v 1. insert", 
                "subescher vwz_u start", "subescher vwz_u end", "vwz_u 1. insert",
                "subescher uv_wz start", "subescher uv_wz end", "uv_wz 1. insert", 
                "subescher uw_vz start", "subescher uw_vz end", "uw_vz 1. insert",
                "subescher vw_uz start", "subescher vw_uz end", "vw_uz 1. insert",]

D = {"0":{"0":"0"}, "subescher start":{}, "subescher end":{}, "1. insertion":{}}
for label in labels:
    if "start" in label:
        D["subescher start"][label] = label.split(" ")[1]
    elif "end" in label:
        D["subescher end"][label] = label.split(" ")[1]
    elif "insert" in label:
        D["1. insertion"][label] = label.split(" ")[0]
print("D:", D)
print(len(labels))


x  {
        '0': {'0': '0'}, 
        'subescher start': {
            'subescher uv start': 'uv', 
            'subescher vw start': 'vw', 
            'subescher uw start': 'uw', 
            'subescher uz start': 'uz', 
            'subescher vz start': 'vz', 
            'subescher wz start': 'wz', 
            'subescher uv_w start': 'uv_w', 
            'subescher uw_v start': 'uw_v', 
            'subescher vw_u start': 'vw_u', 
            'subescher uv_z start': 'uv_z', 
            'subescher uz_v start': 'uz_v', 
            'subescher vz_u start': 'vz_u', 
            'subescher uw_z start': 'uw_z', 
            'subescher uz_w start': 'uz_w', 
            'subescher wz_u start': 'wz_u', 
            'subescher vw_z start': 'vw_z', 
            'subescher vz_w start': 'vz_w', 
            'subescher wz_v start': 'wz_v', 
            'subescher uvw_z start': 'uvw_z', 
            'subescher uvz_w start': 'uvz_w', 
            'subescher uwz_v start': 'uwz_v', 
            'subescher vwz_u start': 'vwz_u', 
            'subescher uv_wz start': 'uv_wz', 
            'subescher uw_vz start': 'uw_vz', 
            'subescher vw_uz start': 'vw_uz'
        }, 
        'subescher end': {
            'subescher uv end': 'uv', 
            'subescher vw end': 'vw', 
            'subescher uw end': 'uw', 
            'subescher uz end': 'uz', 
            'subescher vz end': 'vz', 
            'subescher wz end': 'wz', 
            'subescher uv_w end': 'uv_w', 
            'subescher uw_v end': 'uw_v', 
            'subescher vw_u end': 'vw_u', 
            'subescher uv_z end': 'uv_z', 
            'subescher uz_v end': 'uz_v', 
            'subescher vz_u end': 'vz_u', 
            'subescher uw_z end': 'uw_z', 
            'subescher uz_w end': 'uz_w', 
            'subescher wz_u end': 'wz_u', 
            'subescher vw_z end': 'vw_z', 
            'subescher vz_w end': 'vz_w', 
            'subescher wz_v end': 'wz_v', 
            'subescher uvw_z end': 'uvw_z', 
            'subescher uvz_w end': 'uvz_w', 
            'subescher uwz_v end': 'uwz_v', 
            'subescher vwz_u end': 'vwz_u', 
            'subescher uv_wz end': 'uv_wz', 
            'subescher uw_vz end': 'uw_vz', 
            'subescher vw_uz end': 'vw_uz'
        }, 
        '1. insertion': {
            'uv 1. insert': 'uv', 
            'vw 1. insert': 'vw', 
            'uw 1. insert': 'uw', 
            'uz 1. insert': 'uz', 
            'vz 1. insert': 'vz', 
            'wz 1. insert': 'wz', 
            'uv_w 1. insert': 'uv_w', 
            'uw_v 1. insert': 'uw_v', 
            'vw_u 1. insert': 'vw_u', 
            'uv_z 1. insert': 'uv_z', 
            'uz_v 1. insert': 'uz_v', 
            'vz_u 1. insert': 'vz_u', 
            'uw_z 1. insert': 'uw_z', 
            'uz_w 1. insert': 'uz_w', 
            'wz_u 1. insert': 'wz_u', 
            'vw_z 1. insert': 'vw_z', 
            'vz_w 1. insert': 'vz_w', 
            'wz_v 1. insert': 'wz_v', 
            'uvw_z 1. insert': 'uvw_z', 
            'uvz_w 1. insert': 'uvz_w', 
            'uwz_v 1. insert': 'uwz_v', 
            'vwz_u 1. insert': 'vwz_u', 
            'uv_wz 1. insert': 'uv_wz', 
            'uw_vz 1. insert': 'uw_vz', 
            'vw_uz 1. insert': 'vw_uz'
        }
    }