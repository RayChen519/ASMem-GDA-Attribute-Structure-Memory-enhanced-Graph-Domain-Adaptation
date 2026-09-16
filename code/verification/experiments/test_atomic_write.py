"""Atomic checkpoint replacement failure and recovery contracts."""
import pytest
from data.cache import store


def denied(code=5):
    error = PermissionError('injected Windows access failure')
    error.winerror = code
    return error


@pytest.mark.parametrize('code', [5, 32, 33])
def test_transient_replace_preserves_old_until_commit(tmp_path, monkeypatch, code):
    target = tmp_path / 'checkpoint.pt'
    target.write_bytes(b'old')
    replace = store.os.replace
    calls = []
    sleeps = []
    def flaky(source, destination):
        calls.append(source)
        assert target.read_bytes() == b'old'
        if len(calls) < 3:
            raise denied(code)
        replace(source, destination)
    monkeypatch.setattr(store.os, 'replace', flaky)
    monkeypatch.setattr(store.time, 'sleep', sleeps.append)
    writes = []
    def writer(path):
        writes.append(path)
        path.write_bytes(b'new')
    store.atomic_write(target, writer)
    assert target.read_bytes() == b'new'
    assert len(writes) == 1 and len(calls) == 3
    assert sleeps == [0.05, 0.1]
    assert not list(tmp_path.glob('*.tmp'))


def test_persistent_denial_is_bounded_and_preserves_checkpoint(tmp_path, monkeypatch):
    target = tmp_path / 'checkpoint.pt'
    target.write_bytes(b'old')
    calls = []
    def fail(*args):
        calls.append(args)
        raise denied()
    monkeypatch.setattr(store.os, 'replace', fail)
    monkeypatch.setattr(store.time, 'sleep', lambda _: None)
    with pytest.raises(PermissionError, match='injected') as caught:
        store.atomic_write(target, lambda p: p.write_bytes(b'new'))
    assert len(calls) == 6
    assert '6 attempts' in caught.value.__notes__[0]
    assert target.read_bytes() == b'old'
    assert not list(tmp_path.glob('*.tmp'))


def test_other_io_errors_are_not_retried(tmp_path, monkeypatch):
    def fail(*args):
        raise OSError('disk failure')
    def unexpected_sleep(_):
        pytest.fail('non-Windows error was retried')
    monkeypatch.setattr(store.os, 'replace', fail)
    monkeypatch.setattr(store.time, 'sleep', unexpected_sleep)
    with pytest.raises(OSError, match='disk failure'):
        store.atomic_write(tmp_path / 'checkpoint.pt', lambda p: p.write_bytes(b'new'))
    assert not list(tmp_path.iterdir())


def test_writer_failure_keeps_original(tmp_path):
    target = tmp_path / 'checkpoint.pt'
    target.write_bytes(b'old')
    def writer(path):
        path.write_bytes(b'partial')
        raise ValueError('serialization failed')
    with pytest.raises(ValueError, match='serialization failed'):
        store.atomic_write(target, writer)
    assert target.read_bytes() == b'old'
    assert not list(tmp_path.glob('*.tmp'))
