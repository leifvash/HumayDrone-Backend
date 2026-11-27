from django.core.management.base import BaseCommand

from core.archive import purge_expired_archived_records


class Command(BaseCommand):
    help = "Permanently deletes archived drone flights and survey missions older than 30 days."

    def handle(self, *args, **options):
        summary = purge_expired_archived_records(force=True)
        self.stdout.write(
            self.style.SUCCESS(
                f"Purged {summary['flights_purged']} flight(s), "
                f"{summary['images_purged']} image(s), and "
                f"{summary['surveys_purged']} survey mission(s)."
            )
        )
