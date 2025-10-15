from isaacsim.sensors.physics import ContactSensor
import numpy as np


class LimitSwitchSensor:

    def __init__(self,  prim_path: str):
        self.prim_path = prim_path
        self.limit_switch_sensor = ContactSensor(prim_path=self.prim_path, name="Contact_Sensor")

    def get_force_data(self):
        contacts = self.limit_switch_sensor.get_current_frame()
        force_data = contacts.get('force', 'N/A')
        return force_data

    def get_switch_data(self):
        force = self.get_force_data()
        if force > 100:
            return 1      #到位開關處發發置位
        else:
            return 0      #到位開關沒觸