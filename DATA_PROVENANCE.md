# Data provenance and redistribution notice

The project uses the Steam Video Game and Bundle Data archive obtained from the
[Kaggle mirror](https://www.kaggle.com/datasets/pypiahmad/steam-video-game-and-bundle-data) on
2026-06-05. The downloaded `archive.zip` has SHA-256
`fd23836a5450db0543b4a47b730da1373e20032f8c71d3971e3195392138de0d`.

The primary dataset page is maintained by
[UCSD](https://cseweb.ucsd.edu/~jmcauley/datasets.html#steam_data). The bundle component is described
by Pathak, Gupta, and McAuley, ["Generating and Personalizing Bundle Recommendations on
Steam"](https://doi.org/10.1145/3077136.3080724), SIGIR 2017. Follow the UCSD page's requested
citations when using the data.

The [Kaggle dataset API](https://www.kaggle.com/api/v1/datasets/view/pypiahmad/steam-video-game-and-bundle-data)
reports `Apache 2.0` for mirror version 1, last updated 2023-10-29; I retrieved that metadata on
2026-08-15. The upstream UCSD page provides downloads and requests citation but does not state a
license. The mirror label therefore records the apparent terms of the repackaged archive, but it
does not by itself establish that the uploader could relicense every upstream component. For that
reason:

- raw archives and records are not tracked;
- user identifiers, profile URLs, review text, protected split coordinates, per-user metrics, and
  fitted user parameters are not public artifacts;
- the repository's future software license will not grant rights to third-party source data; and
- permission to redistribute each tracked derived table and figure should be confirmed with the data
  provider and project supervisor before the repository is made public.

See `notes/data_dictionary.md` for fields, checksums, transformations, and interpretation limits.
