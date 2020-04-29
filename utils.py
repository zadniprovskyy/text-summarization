from torch.autograd import Variable


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def to_var(tensor, cuda):
    if cuda:
        return Variable(tensor.cuda())
    else:
        return Variable(tensor)
