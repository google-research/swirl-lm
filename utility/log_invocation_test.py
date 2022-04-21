"""Tests for google3.medical.imaging.utils.log_invocation."""

import datetime
import filecmp
import os
import re
import tempfile
from unittest import mock

from absl import logging
from swirl_lm.utility import log_invocation
from google3.pyglib import gfile
from google3.testing.pybase import googletest
from google3.testing.pybase import parameterized


def ReadFile(filename, timestamp):
  """Reads lines for a given file."""
  if timestamp:
    filename = re.sub(r'\.txt$', f'{timestamp}.txt', filename)
    filename = re.sub(r'\.textpb$', f'{timestamp}.textpb', filename)

  with gfile.Open(filename, 'r') as f:
    lines = f.readlines()
  return [l.rstrip() for l in lines], filename


class LogInvocationTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('WithoutTimestamp', False, ''),
      ('WithTimestamp', True, '_2021_0601_01:02:30:012345'),
  )
  @mock.patch.object(log_invocation, 'datetime')
  def testLogBuilddata(self, add_timestamp, expected_timestamp, mock_datetime):
    """Unit test for `log_builddata`."""
    mock_datetime.datetime.now.return_value = datetime.datetime(
        2021, 6, 1, 1, 2, 30, 12345)

    _, filename = tempfile.mkstemp(dir='/tmp', suffix='.txt')
    new_filename = log_invocation.log_build_data(
        filename, add_timestamp=add_timestamp)

    lines, expected_new_filename = ReadFile(filename, expected_timestamp)
    logging.info('Build data file contents: %s', ''.join(lines))

    self.assertEqual(new_filename, expected_new_filename)
    self.assertEqual(lines[0], 'Build data:')

  @parameterized.named_parameters(
      ('WithoutTimestamp', False, ''),
      ('WithTimestamp', True, '_2020_0601_01:02:30:012345'),
  )
  @mock.patch.object(log_invocation, 'datetime')
  def testLogFlags(self, add_timestamp, expected_timestamp, mock_datetime):
    """Unit test for `log_flags`."""
    mock_datetime.datetime.now.return_value = datetime.datetime(
        2020, 6, 1, 1, 2, 30, 12345)

    _, filename = tempfile.mkstemp(dir='/tmp', suffix='.txt')
    new_filename = log_invocation.log_flags(
        filename, add_timestamp=add_timestamp)

    lines, expected_new_filename = ReadFile(filename, expected_timestamp)
    logging.info('Flags file contents: %s', ''.join(lines))

    self.assertEqual(new_filename, expected_new_filename)
    # First 3 lines.
    self.assertTrue(lines[0].startswith('Command line args: '))
    self.assertEqual(lines[1], '')
    self.assertEqual(lines[2], 'Flags:')

    # Flag lines.
    flag_regex = r'--.+=.*'
    for line in lines[3:]:
      m = re.match(flag_regex, line)
      if not m:
        self.fail(f'Line {line} does not match regex {flag_regex}.')

  @parameterized.named_parameters(
      ('WithoutTimestamp', False, ''),
      ('WithTimestamp', True, '_2021_0601_01:02:30:012345'),
  )
  @mock.patch.object(log_invocation, 'datetime')
  def testLogConfigFile(self, add_timestamp, expected_timestamp, mock_datetime):
    """Unit test for `log_file`."""
    mock_datetime.datetime.now.return_value = datetime.datetime(
        2021, 6, 1, 1, 2, 30, 12345)

    _, filename = tempfile.mkstemp(dir='/tmp', suffix='.textpb')
    if add_timestamp:
      expected_new_filename = re.sub(r'\.textpb$',
                                     f'{expected_timestamp}.textpb', filename)
      self.assertFalse(gfile.Exists(expected_new_filename))
    else:
      expected_new_filename = filename
    logging.info('Config file: %s => %s', filename, expected_new_filename)

    new_filename = log_invocation.log_file(
        filename, filename, add_timestamp=add_timestamp)

    self.assertEqual(expected_new_filename, new_filename)
    self.assertTrue(gfile.Exists(expected_new_filename))

    self.assertTrue(filecmp.cmp(filename, expected_new_filename, shallow=False))

  @parameterized.named_parameters(
      ('WithoutTimestamp00', False, 0,
       (Exception, gfile.GOSError, gfile.GOSError)),
      ('WithoutTimestamp01', False, 0,
       (gfile.GOSError, Exception, gfile.GOSError)),
      ('WithoutTimestamp02', False, 0,
       (gfile.GOSError, gfile.GOSError, Exception)),
      ('WithTimestamp00', True, 3, (Exception, gfile.GOSError, gfile.GOSError)),
      ('WithTimestamp01', True, 3, (gfile.GOSError, Exception, gfile.GOSError)),
      ('WithTimestamp02', True, 3, (gfile.GOSError, gfile.GOSError, Exception)),
  )
  @mock.patch.object(log_invocation, 'datetime')
  @mock.patch.object(log_invocation, 'gfile')
  def testLogConfigFileWithExceptions(self, add_timestamp, expected_call_count,
                                      mock_gfile_copy_side_effect, mock_gfile,
                                      mock_datetime):
    """Unit test for `log_file`."""
    mock_datetime.datetime.now.return_value = datetime.datetime(
        2021, 6, 1, 1, 2, 30, 12345)
    mock_gfile.Copy.side_effect = mock_gfile_copy_side_effect

    _, filename = tempfile.mkstemp(dir='/tmp', suffix='.textpb')
    with gfile.Open(filename, 'wb') as f:
      f.write(os.urandom(123))

    new_filename = log_invocation.log_file(
        filename, filename, add_timestamp=add_timestamp)

    self.assertEqual(mock_gfile.Copy.call_count, expected_call_count)
    self.assertEqual(mock_datetime.datetime.now.call_count, expected_call_count)
    if add_timestamp:
      # Failed to write.
      self.assertIsNone(new_filename)
    else:
      # No op.
      self.assertEqual(filename, new_filename)

  @parameterized.named_parameters(
      ('NoOp', False, True),
      ('WithExportDir', True, False),
  )
  def testLogJob(self, make_dir, expect_nones):
    """Tests for `log_job` for combined outputs."""
    if make_dir:
      tmp_dir = tempfile.mkdtemp()
      _, from_filename = tempfile.mkstemp(dir=tmp_dir, suffix='.txt')
    else:
      tmp_dir = ''
      from_filename = ''

    job = log_invocation.log_job(tmp_dir, from_filename)

    if expect_nones:
      self.assertIsNone(job)
    else:
      self.assertLen(job, 3)
      for i in range(3):
        self.assertIsInstance(job[i], str)

if __name__ == '__main__':
  googletest.main()
