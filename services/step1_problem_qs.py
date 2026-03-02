from django.db.models import Exists, OuterRef, Q


def build_problem_qs(*, day: str):
    from ht.apps.reports.models import LandActivity, LandJob, MTurkLock, SummaryAPICache

    la_exists = LandActivity.objects.filter(summaryapicache_id=OuterRef("pk"))

    la_has_creator = LandActivity.objects.filter(
        summaryapicache_id=OuterRef("pk"),
        creator_user__isnull=False,
    )

    turk_lock_exists = MTurkLock.objects.filter(summaryapicache_id=OuterRef("pk"))

    landjob_polygon_exists = LandJob.objects.with_deleted_unapproved().filter(
        activity_id=OuterRef("pk"),
        deleted_at__isnull=True,
        polygon__isnull=False,
    )

    return (
        SummaryAPICache.objects.filter(day=day)
        .annotate(
            has_land_activity=Exists(la_exists),
            has_land_activity_creator=Exists(la_has_creator),
            has_turk_lock=Exists(turk_lock_exists),
            has_polygon=Exists(landjob_polygon_exists),
        )
        .filter(Q(has_land_activity=False) | Q(has_polygon=False))
        .filter(has_turk_lock=False, has_land_activity_creator=False)
    )
