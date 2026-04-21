"""tracerCC runtime — drop a fitted routing policy into your OpenAI client.

See ``tracercc.runtime.router.Router`` for the main surface. Typical wiring
in a Hermes-style agent gateway:

    from tracercc.runtime import Router

    router = Router.from_file("./tracer_policy.json", default_model="gpt-5.2")

    # before every openai.chat.completions.create call:
    model_to_use, rule_id = router.route(messages=messages, tools=tools)
    result = openai.chat.completions.create(model=model_to_use, messages=messages, tools=tools)
    router.record(rule_id, result)   # optional: grows the refit dataset

``Router.route`` is O(n_rules). No network, no extra API calls. Loads a
policy JSON that ``tracercc`` produced from your past sessions.
"""

from .router import Router, load_policy, apply_policy

__all__ = ["Router", "load_policy", "apply_policy"]
