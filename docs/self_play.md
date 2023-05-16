# Self-Play

Turing point includes also a template class for self-play, SelfPlay.
SelfPlay in Turing point has its stand-alone interface and is not related to the participants/assembly main theme of the package.

You can of course use both the SelfPlay template and participants/assembly launches as works for you in your application.

For a SelfPlay you need to implement a few methods (abstract in the template class SelfPlay), and also to provide a couple of callables in the call to 'launch'.

Abstract methods to implement:

- fetch_agent_to_train
- save_agent
- fetch_opponent
- train_against_agent
- evaluate_agent

Callables to provide as arguments to 'launch':

- should_stop
- should_save

Below is the code of 'launch':

``` py
def launch(self,
            should_stop: Callable[[], bool],
            should_save: Callable[[], bool]) -> Quality:
  agent_to_train = self.fetch_agent_to_train()
  while not should_stop():
    opponent_agent = self.fetch_opponent()
    self.train_against_agent(agent_to_train, opponent_agent)
    if should_save():
      self.save_agent(agent_to_train)
  return self.evaluate_agent(agent_to_train)
```