# Stockfish notice

The CPU workspace installs Debian Trixie's unmodified `stockfish` package
(version 17-1 for the pinned `python:3.12-slim` Trixie base image). Stockfish is licensed
under the GNU General Public License version 3. The installed notice is
`/usr/share/doc/stockfish/copyright`; the complete license is `/usr/share/common-licenses/GPL-3`.

Corresponding source is available from [Debian's Stockfish 17-1 source package](https://sources.debian.org/src/stockfish/17-1/)
and the [upstream Stockfish repository](https://github.com/official-stockfish/Stockfish/tree/sf_17).
The Debian package supplies the engine binary in the workspace image; no engine binary
is checked into this repository.
