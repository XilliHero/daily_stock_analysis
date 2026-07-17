import type React from 'react';
import { useState } from 'react';
import type { ParsedApiError } from '../../api/error';
import { isParsedApiError } from '../../api/error';
import { useAuth } from '../../hooks';
import { Button, Input } from '../common';
import { SettingsAlert } from './SettingsAlert';
import { SettingsSectionCard } from './SettingsSectionCard';

export const ChangePasswordCard: React.FC = () => {
  const { changePassword } = useAuth();
  const [currentPassword, setCurrentPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [newPasswordConfirm, setNewPasswordConfirm] = useState('');
  
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | ParsedApiError | null>(null);
  const [success, setSuccess] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setSuccess(false);

    if (!currentPassword.trim()) {
      setError('Enter the current password');
      return;
    }
    if (!newPassword.trim()) {
      setError('Enter a new password');
      return;
    }
    if (newPassword.length < 6) {
      setError('New password: at least 6 characters');
      return;
    }
    if (newPassword !== newPasswordConfirm) {
      setError('The two new passwords don\'t match');
      return;
    }

    setIsSubmitting(true);
    try {
      const result = await changePassword(currentPassword, newPassword, newPasswordConfirm);
      if (result.success) {
        setSuccess(true);
        setCurrentPassword('');
        setNewPassword('');
        setNewPasswordConfirm('');
        setTimeout(() => setSuccess(false), 4000);
      } else {
        setError(result.error ?? 'Change failed');
      }
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <SettingsSectionCard
      title="Change password"
      description="Update the admin login password. After a successful change, log in with the new password."
    >
      <form onSubmit={handleSubmit} className="space-y-3">
        <div className="grid gap-4 md:grid-cols-2">
          <div className="space-y-3">
            <Input
              id="change-pass-current"
              type="password"
              allowTogglePassword
              iconType="password"
              label="Current password"
              placeholder="Enter the current password"
              value={currentPassword}
              onChange={(e) => setCurrentPassword(e.target.value)}
              disabled={isSubmitting}
              autoComplete="current-password"
            />
          </div>

          <div className="space-y-3">
            <Input
              id="change-pass-new"
              type="password"
              allowTogglePassword
              iconType="password"
              label="New password"
              hint="At least 6 characters."
              placeholder="Enter a new password"
              value={newPassword}
              onChange={(e) => setNewPassword(e.target.value)}
              disabled={isSubmitting}
              autoComplete="new-password"
            />
          </div>
        </div>

        <div className="space-y-3 md:max-w-md">
          <Input
            id="change-pass-confirm"
            type="password"
            allowTogglePassword
            iconType="password"
            label="Confirm new password"
            placeholder="Re-enter the new password"
            value={newPasswordConfirm}
            onChange={(e) => setNewPasswordConfirm(e.target.value)}
            disabled={isSubmitting}
            autoComplete="new-password"
          />
        </div>

        {error
          ? isParsedApiError(error)
            ? <SettingsAlert title="Change failed" message={error.message} variant="error" className="!mt-3" />
            : <SettingsAlert title="Change failed" message={error} variant="error" className="!mt-3" />
          : null}
        {success ? (
          <SettingsAlert title="Changed successfully" message="Admin password updated." variant="success" />
        ) : null}

        <Button type="submit" variant="primary" isLoading={isSubmitting}>
          Save new password
        </Button>
      </form>
    </SettingsSectionCard>
  );
};
