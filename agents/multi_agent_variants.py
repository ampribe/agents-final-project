from agents.multi_agent import MultiAgent


class MultiAgentFull(MultiAgent):
    def __init__(self, model: str, task: str, max_steps: int = 30, logger=None):
        super().__init__(
            model=model,
            task=task,
            max_steps=max_steps,
            logger=logger,
            enable_researcher=True,
            enable_tester=True,
        )


class MultiAgentNoResearcher(MultiAgent):
    def __init__(self, model: str, task: str, max_steps: int = 30, logger=None):
        super().__init__(
            model=model,
            task=task,
            max_steps=max_steps,
            logger=logger,
            enable_researcher=False,
            enable_tester=True,
        )


class MultiAgentNoTester(MultiAgent):
    def __init__(self, model: str, task: str, max_steps: int = 30, logger=None):
        super().__init__(
            model=model,
            task=task,
            max_steps=max_steps,
            logger=logger,
            enable_researcher=True,
            enable_tester=False,
        )


class MultiAgentCoderOnly(MultiAgent):
    def __init__(self, model: str, task: str, max_steps: int = 30, logger=None):
        super().__init__(
            model=model,
            task=task,
            max_steps=max_steps,
            logger=logger,
            enable_researcher=False,
            enable_tester=False,
        )
