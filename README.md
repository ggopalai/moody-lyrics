---
title: Moody Lyrics
emoji: 👀
colorFrom: gray
colorTo: indigo
sdk: docker
pinned: false
license: mit
---


# Moody Lyrics

Song recommendation website.

## High-Level Functional Features

- Associates a song with a mood based on the lyrics.
- Lists out similar songs. (TODO)

## Eventual Non-Functional Features

- Scale - 1 million reqs/second
- Availability - Limited downtime

### High Level Architecture

- A simple web app that takes in song name and artist.
- A backend server that -
  - Fetches the lyrics from genius.com
  - Passes the lyrics into the model to assign a mood.
  - Uses this mood to fetch similar songs. (TODO)
- Backend served and scaled by TFX (TODO)

### Tech Stack

- Front-end - React (TODO)
- Rest of the stuff in Python.
- Database - TBD

## TODO
- Deploy the simple flask app
- Added functionality: Interpet model prediction
- Added functionality: Store predictions
- Added functionality: Validation by user