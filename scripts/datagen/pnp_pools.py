"""Fixed MolmoSpaces PnP collection pools sourced from successful MolmoData demos.

Each pool fixes one task identity. The sampler still discovers and caches the
pickup object's original supporting geometry when the selected house loads.
Pools are candidates until they pass the current noise-off, zero-retry smoke.
"""

PNP_POOLS = {
    "stopwatch_bowl25": {
        "scene_dataset": "ithor",
        "data_split": "train",
        "house_inds": 10,
        "pickup_obj_name": "objastopwatch_ab795908c0874284a01556b71ac8f34a_1_0_3",
        "fixed_place_receptacle_uid": "Bowl_25",
        "source": "controlled_gate2_candidate",
        "status": "candidate",
    },
    "molmodata_candle_bowl_1379": {
        "scene_dataset": "procthor-objaverse",
        "data_split": "val",
        "house_inds": 1379,
        "pickup_obj_name": "objacandle_857d3f1a93b54b25bcc14aab9203346e_1_0_3",
        "fixed_place_receptacle_uid": "27dd012c10e64049b8ca900fe86f077f",
        "source": "MolmoData_val_00000_seed_0",
        "status": "candidate",
    },
    "molmodata_potato_bowl_1716": {
        "scene_dataset": "procthor-objaverse",
        "data_split": "val",
        "house_inds": 1716,
        "pickup_obj_name": "Irishpotato_4ccdc5ebde4d6fee07ff9eefb0b60cfb_1_0_2",
        "fixed_place_receptacle_uid": "5c5c3b9ae7874b709c10ac57dad33195",
        "source": "MolmoData_val_00000_seed_7",
        "status": "candidate",
    },
    "molmodata_pocketwatch_bowl_2305": {
        "scene_dataset": "procthor-objaverse",
        "data_split": "val",
        "house_inds": 2305,
        "pickup_obj_name": "objapocketwatch_dbb40c6f9226401f88230d46a12fa50f_1_0_3",
        "fixed_place_receptacle_uid": "0b0079b036c845139d5823255a34da7a",
        "source": "MolmoData_val_00000_seed_13",
        "status": "candidate",
    },
    "molmodata_pocketwatch2_bowl_2305": {
        "scene_dataset": "procthor-objaverse",
        "data_split": "val",
        "house_inds": 2305,
        "pickup_obj_name": "objapocketwatch_2ec1ab458bcd4ea59236bb58bc3707dd_1_0_3",
        "fixed_place_receptacle_uid": "0b0079b036c845139d5823255a34da7a",
        "source": "MolmoData_val_00000_seed_14",
        "status": "candidate",
    },
    "molmodata_potato_bowl_3080": {
        "scene_dataset": "procthor-objaverse",
        "data_split": "val",
        "house_inds": 3080,
        "pickup_obj_name": "Irishpotato_e58a7a0c37cb0ef9c8155d56cc319ccf_1_0_2",
        "fixed_place_receptacle_uid": "10578ce7eca64714b260b142553f34f5",
        "source": "MolmoData_val_00000_seed_33",
        "status": "candidate",
    },
    "molmodata_phone_bowl_3348": {
        "scene_dataset": "procthor-objaverse",
        "data_split": "val",
        "house_inds": 3348,
        "pickup_obj_name": "cellulartelephone_1fa394556eb37156656d60fc604d4668_2_0_6",
        "fixed_place_receptacle_uid": "5c5c3b9ae7874b709c10ac57dad33195",
        "source": "MolmoData_val_00000_seed_35",
        "status": "candidate",
    },
    "molmodata_soap_tray_3536": {
        "scene_dataset": "procthor-objaverse",
        "data_split": "val",
        "house_inds": 3536,
        "pickup_obj_name": "soapdispenser_23e99c616e7c0667edeeacb9669a5e0b_1_0_2",
        "fixed_place_receptacle_uid": "f486de07c6f347588bcd3d22456f328d",
        "source": "MolmoData_val_00000_seed_41",
        "status": "candidate",
    },
    "molmodata_phone_bowl_3575": {
        "scene_dataset": "procthor-objaverse",
        "data_split": "val",
        "house_inds": 3575,
        "pickup_obj_name": "cellulartelephone_43b2396be97b1322a94e498589888114_1_0_2",
        "fixed_place_receptacle_uid": "bfae6fa9a7f64b9eb16f57c05553c612",
        "source": "MolmoData_val_00000_seed_43",
        "status": "candidate",
    },
    "molmodata_digitalwatch_bowl_4069": {
        "scene_dataset": "procthor-objaverse",
        "data_split": "val",
        "house_inds": 4069,
        "pickup_obj_name": "objadigitalwristwatch_0518a9ae9cd14ae89efffa4213a8cce4_1_0_7",
        "fixed_place_receptacle_uid": "4c75946736424e5397f4eadd336e0bf1",
        "source": "MolmoData_val_00000_seed_52",
        "status": "candidate",
    },
    "molmodata_cup_receptacle_4319": {
        "scene_dataset": "procthor-objaverse",
        "data_split": "val",
        "house_inds": 4319,
        "pickup_obj_name": "cup_941e8ae5e6392530fc6f44a310d0ba5a_1_0_6",
        "fixed_place_receptacle_uid": "0500b0d9ef6f40dd96591d7b354c9ede",
        "source": "MolmoData_val_00000_seed_57",
        "status": "candidate",
    },
    "molmodata_egg_bowl_4389": {
        "scene_dataset": "procthor-objaverse",
        "data_split": "val",
        "house_inds": 4389,
        "pickup_obj_name": "egg_45a3d68915c9f19164541ddf4da76856_1_0_8",
        "fixed_place_receptacle_uid": "9ed70e7516894535a9e208904f8db364",
        "source": "MolmoData_val_00000_seed_58",
        "status": "candidate",
    },
    "molmodata_tomato_bowl_4519": {
        "scene_dataset": "procthor-objaverse",
        "data_split": "val",
        "house_inds": 4519,
        "pickup_obj_name": "tomato_8c4ab0ebeae63b9cd60e6686b8602751_1_0_8",
        "fixed_place_receptacle_uid": "1ef337c5cfda4d368ceb154c0e9fae0e",
        "source": "MolmoData_val_00000_seed_62",
        "status": "candidate",
    },
    "molmodata_bottle_tray_4572": {
        "scene_dataset": "procthor-objaverse",
        "data_split": "val",
        "house_inds": 4572,
        "pickup_obj_name": "objabottle_745d282ade2c4b4c8a2b6d8c27cd9112_1_0_8",
        "fixed_place_receptacle_uid": "0500b0d9ef6f40dd96591d7b354c9ede",
        "source": "MolmoData_val_00000_seed_64",
        "status": "candidate",
    },
    "molmodata_mobilephone_pot_4880": {
        "scene_dataset": "procthor-objaverse",
        "data_split": "val",
        "house_inds": 4880,
        "pickup_obj_name": "objavintagemobilephone_e1a072af323f4d358bad054195353b96_2_0_6",
        "fixed_place_receptacle_uid": "93ef7da87b1e43fbb47f3b26b927f777",
        "source": "MolmoData_val_00000_seed_70",
        "status": "candidate",
    },
    "molmodata_soap_bowl_5134": {
        "scene_dataset": "procthor-objaverse",
        "data_split": "val",
        "house_inds": 5134,
        "pickup_obj_name": "soapdispenser_2b480aa666c86f6fbe993d7230e43625_1_0_3",
        "fixed_place_receptacle_uid": "80f09dd8b3d34005b3625823091978af",
        "source": "MolmoData_val_00000_seed_73",
        "status": "candidate",
    },
}


def get_pool(name: str) -> dict:
    try:
        return dict(PNP_POOLS[name])
    except KeyError as exc:
        names = ", ".join(sorted(PNP_POOLS))
        raise ValueError(f"Unknown PnP pool {name!r}. Available pools: {names}") from exc
