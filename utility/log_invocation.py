"""Library to log invocation-related information for any jobs.

Example Usage:

def main(argv):
  export_dir = <dirname>
  config_filename = <filename>

  log_invocation.log_file(export_dir, config_filename)

  # Actual binary.
"""

import datetime
import os
import re
import sys
import textwrap
from typing import Optional, Text, Tuple

from absl import flags
from absl import logging
from google3.pyglib import build_data
from google3.pyglib import gfile

FLAGS = flags.FLAGS


FILENAME_FOR_BUILD_DATA = 'build_data.txt'
FILENAME_FOR_FLAGS = 'flags.txt'

_MAX_ATTEMPTS = 3


def get_timestamp(fmt: Text = '%Y_%m%d_%H:%M:%S:%f') -> Text:
  """Gets current timestamp, could be used as filename so that it's ~unique."""
  return datetime.datetime.now().strftime(fmt)


def get_filename_with_timestamp(filename: Text) -> Text:
  """Gets filename with timestamp."""
  timestamp = get_timestamp()
  prefix, suffix = os.path.splitext(filename)
  return f'{prefix}_{timestamp}{suffix}'


def _export_to_file(filename: Text,
                    contents: Text,
                    add_timestamp: bool = True) -> Text:
  """Writes the given contents to the given filename via gfile."""
  if add_timestamp:
    filename = get_filename_with_timestamp(filename)

  dir_name = os.path.dirname(filename)
  if not gfile.Exists(dir_name):
    gfile.MakeDirs(dir_name)

  with gfile.Open(filename, 'w') as f:
    f.write(contents)
  logging.info('Wrote to %s.', filename)
  return filename


def log_build_data(filename: Text, add_timestamp: bool = True) -> Text:
  """Writes builddata information to the given filename."""
  data = textwrap.dedent("""\
      Build data:
      {builddata}
  """).format(builddata=build_data.BuildData())

  return _export_to_file(filename, data, add_timestamp=add_timestamp)


def log_flags(filename: Text, add_timestamp: bool = True) -> Text:
  """Writes flag information to the given filename."""
  output = ['Command line args: %s' % ' '.join(sys.argv), '', 'Flags:']
  for name, value in sorted(FLAGS.flag_values_dict().items()):
    output.append(f'--{name}={value}')
  data = '\n'.join(output)

  return _export_to_file(filename, data, add_timestamp=add_timestamp)


def log_file(from_filename: Text,
             filename: Text,
             add_timestamp: bool = True) -> Optional[Text]:
  """Copies file to another file potentially with a timestamp."""
  for i in range(_MAX_ATTEMPTS):
    if add_timestamp:
      to_filename = get_filename_with_timestamp(filename)
    else:
      to_filename = filename

    if to_filename == from_filename:
      logging.info('No op for copy: %s.', to_filename)
      return to_filename

    try:
      gfile.Copy(from_filename, to_filename)
      logging.info('Wrote to %s.', to_filename)
      return to_filename
    except Exception as e:  # pylint: disable=broad-except
      logging.warning('Unable to write to %s [%d/%d]: %s.', to_filename, i,
                      _MAX_ATTEMPTS, str(e))

  logging.warning('Unable to write to %s, after %d attempts.', filename,
                  _MAX_ATTEMPTS)
  return None


def log_job(
    export_dir: Text,
    from_filename: Text,
    filename: Optional[Text] = None,
    add_timestamp: bool = True,
) -> Optional[Tuple[Text, Text, Optional[Text]]]:
  """Writes job info available.

  It tries to write down the following info in the specified export directory:
  1. Build data
  2. Flags
  3. A file, which is usually a config file.

  Args:
    export_dir: Export directory for all job info. If empty, it's a no op.
    from_filename: Source (config) file to read from.
    filename: Optional destination file to write to, under `export_dir`.
    add_timestamp: Whether to add a timestamp for the destination file, in case
      a job is preempted, and re-run multiple times.

  Returns:
    Combined return from the 3 separate sub jobs.
  """
  if not export_dir:
    logging.info('Skip logging as `export_dir` is not given.')
    return None

  if filename is None:
    filename = os.path.join(export_dir, re.sub(os.path.sep, '.', from_filename))

  return (
      log_build_data(os.path.join(export_dir, FILENAME_FOR_BUILD_DATA)),
      log_flags(os.path.join(export_dir, FILENAME_FOR_FLAGS)),
      log_file(from_filename, filename, add_timestamp=add_timestamp),
  )
