"""Load this experiment's trusted NEAT checkpoints from any Python entry point."""
import gzip
import pickle
from .evolve import AuditGenome,plain_tanh


class EvolutionUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        # Runs launched with `python -m` recorded these symbols under __main__.
        if module=='__main__' and name in {'AuditGenome','plain_tanh'}:
            return {'AuditGenome':AuditGenome,'plain_tanh':plain_tanh}[name]
        return super().find_class(module,name)


def load_evolution_checkpoint(path):
    """Return population, Python random state and accounting; load trusted files only.

    The checkpoint was written after reproduction. Its population is the next
    generation, while generation_*.json.gz contains the evaluated generations.
    This loader does not launch training or change global random state.
    """
    with gzip.open(path,'rb') as stream:return EvolutionUnpickler(stream).load()
