import carb.input
import omni.appwindow

class VN_KeyBoard:
    def __init__(self):
        # get keyboard
        keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        # subscription
        keyboard_event_sub = (carb.input.acquire_input_interface()
                              .subscribe_to_keyboard_events(keyboard, self.on_keyboard_event))

    @staticmethod
    def on_keyboard_event(self, event):

        # print(f"Input event: {event.device} {event.input} {event.keyboard} {event.modifiers} {event.type}")
        # key UP pressed/released
        if event.input == carb.input.KeyboardInput.UP:
            if event.type == carb.input.KeyboardEventType.KEY_PRESS or event.type == carb.input.KeyboardEventType.KEY_REPEAT:
                pass

        # key DOWN pressed/released
        elif event.input == carb.input.KeyboardInput.DOWN:
            if event.type == carb.input.KeyboardEventType.KEY_PRESS or event.type == carb.input.KeyboardEventType.KEY_REPEAT:
                pass

        # key RIGHT pressed/released
        elif event.input == carb.input.KeyboardInput.RIGHT:
            if event.type == carb.input.KeyboardEventType.KEY_PRESS or event.type == carb.input.KeyboardEventType.KEY_REPEAT:
                pass

        # key LEFT pressed/released
        elif event.input == carb.input.KeyboardInput.LEFT:
            if event.type == carb.input.KeyboardEventType.KEY_PRESS or event.type == carb.input.KeyboardEventType.KEY_REPEAT:
                pass

        # key i pressed/released
        elif event.input == carb.input.KeyboardInput.I:
            if event.type == carb.input.KeyboardEventType.KEY_PRESS or event.type == carb.input.KeyboardEventType.KEY_REPEAT:
                pass

        # key k pressed/released
        elif event.input == carb.input.KeyboardInput.K:
            if event.type == carb.input.KeyboardEventType.KEY_PRESS or event.type == carb.input.KeyboardEventType.KEY_REPEAT:
                pass

        # key S pressed/released
        elif event.input == carb.input.KeyboardInput.S:
            if event.type == carb.input.KeyboardEventType.KEY_PRESS:
                pass