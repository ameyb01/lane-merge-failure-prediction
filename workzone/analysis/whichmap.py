import carla

client = carla.Client("localhost", 2000); client.set_timeout(20.0)
world = client.get_world()
print("current map :", world.get_map().name)
print("available   :", [m.split('/')[-1] for m in client.get_available_maps()])
