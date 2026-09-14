"""P7/P8 integration prerequisite. No formal training is launched here."""
from verification.gate.run import require_pass


def require_integration(report_path):
    """Fail closed on missing, incomplete, failed, modified or stale P6 evidence.

    This is one prerequisite, not authorization to omit development-config,
    registry or immutable-run-manifest locking required by P8.
    """
    return require_pass(report_path)


def check_cli(args):
    """All five non-smoke CLI stages require current integration evidence."""
    if not args.smoke:
        path = getattr(args, 'integration_report', None)
        if path is None:
            raise PermissionError('Non-smoke training requires --integration-report with current P6 PASS')
        return require_integration(path)
