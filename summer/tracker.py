class ActionType:
    STRATIFY = "stratify"
    ADJUST_POP_SPLIT = "adjust_pop_split"


class TrackedAction:
    def __init__(self, action_type: ActionType, kwargs):
        self.action_type = action_type
        self.kwargs = kwargs


class ModelBuildTracker:
    def __init__(self):
        self.all_actions = []

    def append_action(self, action_type: ActionType, **kwargs):
        action = TrackedAction(action_type, kwargs)
        self.all_actions.append(action)
