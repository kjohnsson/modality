class LinkedList(object):
    def __init__(self, data=None):
        self._len = 0
        if not data is None:
            self.append(data)

    def __len__(self):
        return self._len

    def __iter__(self):
        return LinkedListIterator(self)

    def append(self, data):
        if len(self) > 0:
            self.last.next = ListItem(data)
        else:
            self.first = ListItem(data)
            self.before_first = ListItem(None)
            self.before_first.next = self.first
            self.last = self.first
            self._len += 1


class ListItem(object):
    def __init__(self, data):
        self.data = data

    # @property
    # def next(self):
    #     try:
    #         return self._next
    #     except AttributeError:
    #         raise OutsideListError

    # @next.setter
    # def next(self, next):
    #     self._next = next


class LinkedListIterator(object):
    def __init__(self, linked_list):
        self.curr = linked_list.before_first

    def __iter__(self):
        return self

    def next(self):
        try:
            self.curr = self.curr.next
        except AttributeError:
            raise StopIteration
        return self.curr


class LinkedIntervals(LinkedList):

    def split(self, i, insert_interval):
        for j, interval in enumerate(self):
            if j == i:
                try:
                    next = interval.next
                except AttributeError:
                    no_next = True
                else:
                    no_next = False
                imin, imax = interval.data
                insertmin, insertmax = insert_interval
                interval.data = imin, insertmin
                interval.next = ListItem((insertmin, insertmax))
                interval.next.next = ListItem((insertmax, imax))
                if not no_next:
                    interval.next.next.next = next
                self._len += 2
                return