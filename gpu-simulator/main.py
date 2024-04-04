import accel_sim


accel_sim_instance = accel_sim.accel_sim_framework('gpgpusim.config', 'kernelslist.g')
accel_sim_instance.simulation_loop()

print("GPGPU-Sim: *** simulation thread exiting ***")
print("GPGPU-Sim: *** exit detected ***")
