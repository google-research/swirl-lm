# Copyright 2025 The swirl_lm Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Library for common file IO operations."""

import csv
import logging
import os.path

from google.protobuf import message
import numpy as np
from swirl_lm.utility import file_pb2
import tensorflow as tf



def parse_csv_file(
    path: str,
) -> dict[str, np.ndarray]:
  """Reads and returns the columns of the input csv file (with header row).

  Args:
    path: Path of the csv file.

  Returns:
    A dictionary mapping column names to a `tf.Tensor` of the column values.
  """
  out = {}
  with tf.io.gfile.GFile(path, 'r') as f:
    for i, row in enumerate(csv.DictReader(f)):
      for key, value in row.items():
        out.setdefault(key, []).append(float(value))
      assert all(
          len(values) == i + 1 for values in out.values()
      ), f'Missing values in {path} while processing {row}.'
  return {
      key: np.array(values, dtype=np.float32) for key, values in out.items()
  }


def _load_from_path(path: str) -> str:
  """Attempts to load file pointed to by `path`.

  Args:
    path: The path to the file to load.

  Returns:
    The contents of the file.
  """
  with tf.io.gfile.GFile(path, 'r') as f:
    contents = f.read()
  logging.info('Loaded file `%s` from file system.', path)
  return contents


def load_file(file_spec: file_pb2.File) -> str:
  """Loads file."""
  return _load_from_path(file_spec.path)


def find_referred_files(msg: message.Message) -> list[file_pb2.File]:
  """Finds and returns file_pb2.File messages inside `msg`."""
  if isinstance(msg, file_pb2.File):
    return [msg]

  out = []
  for descriptor, value in msg.ListFields():
    # Visit all message fields inside `msg` and ignore scalar fields because
    # file_pb2.File can only be in a message field or contained by some other
    # message field.
    if descriptor.type == descriptor.TYPE_MESSAGE:
      # A message field can be repeated or not. If it is repeated, then visit
      # all the elements. Otherwise just visit the field. Note that we have to
      # use descriptor.type to check for message type instead of
      # isinstance(value, message.Message) because repeated fields are in
      # container classes that are not message.Messages.
      if descriptor.label == descriptor.LABEL_REPEATED:
        for element in value:
          assert isinstance(element, message.Message), (
              f'Expected type message.Message, got `{type(element)}` for '
              f'`{element}`.')
          out.extend(find_referred_files(element))
      else:
        assert isinstance(value, message.Message), (
            f'Expected type message.Message, got `{type(value)}` for '
            f'`{value}`.')
        out.extend(find_referred_files(value))
  return out


def copy_files(files: list[file_pb2.File], output_dir: str) -> None:
  """Copies the contents of each file in `files` to `output_dir.

  The output filenames match the basenames of files but the directory paths
  are ignored. For example given 'a.txt', 'x/b.txt' and 'y/c.txt', the output
  directory will contain 'a.txt', 'b.txt' and 'c.txt'. If there are duplicate
  basenames (which should be rare in practice), they will be made unique by
  appending consecutive integers. For example given 'x/a.txt' and 'y/a.txt',
  the output directory will contain 'a.txt' and 'a.txt.1'.

  Args:
    files: List of input files to copy.
    output_dir: Target directory.
  """
  count_by_basename = {}
  seen_paths = set()

  def unique_basename(path):
    basename = os.path.basename(path)
    count = count_by_basename.get(basename, 0)
    if count == 0:
      unique_name = basename
    else:
      unique_name = f'{basename}.{count}'
    count_by_basename[basename] = count + 1
    return unique_name

  for source_file in files:
    if source_file.path in seen_paths:
      continue
    seen_paths.add(source_file.path)
    target_name = unique_basename(source_file.path)
    with tf.io.gfile.GFile(os.path.join(output_dir, target_name), 'w') as f:
      f.write(load_file(source_file))
